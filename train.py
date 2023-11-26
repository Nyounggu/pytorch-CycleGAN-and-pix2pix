"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
if __name__ == '__main__':
    opt = TrainOptions().parse()   # TrainOptions 인스턴스를 만들고, 명령줄 인수를 구문분석하여 다양한 훈련 옵션 및 설정을 얻음
    dataset = create_dataset(opt)  # Opt에 전달된 구문을 바탕으로 데이터셋 생성
    dataset_size = len(dataset)    #데이터셋 내의 이미지 갯수가 곧 데이터셋의 크기
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # Opt옵션에 따라 모델 생성
    model.setup(opt)               # 네트워크 구조를 로드하고 출력하는 작업과 훈련을 위한 스케줄러를 생성함
    visualizer = Visualizer(opt)   # 시각화 도구 생성
    total_iters = 0                

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    #opt에서 받은 n_epochs 와 n_epoch_decay를 더하여 반복루프를 시작함
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # 시각화 도구 리셋, 최소 한번의 에폭마다 결과를 저장함
        model.update_learning_rate()    # 에폭 시작 시 학습률을 조절함, 
        for i, data in enumerate(dataset):  # 에폭에서 내에서 데이터셋의 데이터를 가져오는 루프 실행
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # 데이터를 데이터셋에서 풀고 데이터를 전처리하여 input
            model.optimize_parameters()   # 모델의 매개변수를 최적화하고 손실 함수를 게산하며, gradient를 계산하고 네트워크 가중치를 업데이트 함

            if total_iters % opt.display_freq == 0:   # 결과를 HTML파일에 저장 
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()    #모델의 시각화 계산, 모델의 출력을 시각적으로 표현하는데 사용되는 이미지나 그래프 생성           visualizer.display_current_results(model.get_current_visuals(), epoch, save_result) #현재 결과를 시각화 하고, save_result가 참인 경우 HTML 파일에 결과를 저장
            if total_iters % opt.print_freq == 0:    # 일정 주기 마다 결과를 화면에 출력하기 위한 if문,print training losses and save logging information to the disk
                losses = model.get_current_losses() #모델의 손실율 가져옴
                t_comp = (t1ime.time() - iter_start_time) / opt.batch_size 
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)#에폭,에폭 내 반복 횟수, 현재 손실, 계산시간 및 데이터 로딩시간을 인수로 받음

                if opt.display_id > 0: 
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses) 
#구문에서 display_id가 0이하라면 화면에 표시하지 않는다. 손실 값을 그래프로 그리는 visualizer

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations 최신 모델을 주기적으로 저장하기위한 if문
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix) #모델의 네트워크 가중치를 저장하는데 사용되는 메서드, 모델의 현재 상태를 저장

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
