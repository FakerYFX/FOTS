import warnings
class DefaultConfig(object):

    initial_epoch = 0
    epoch_num = 30000
    lr = 1e-3
    decay = 5e-4
    use_gpu = True
    batch_size = 64
    num_workers = 10
    optmizer = 'RMSprop' # RMSprop,Adam # SGD, SGD
    betas = (0.5,0.999)
    epsilon = 1e-4
    shrink_side_ratio = 0.6
    shrink_ratio = 0.2
    model = 'FOTS'

    patience = 2
    load_weights = False
    lambda_inside_score_loss = 4.0
    lambda_side_vertex_code_loss = 1.0
    lambda_side_vertex_coord_loss = 1.0

    load_model_path = "checkpoints/model.pth"
    save_path = "save_model/"


    total_img = 16243

    validation_split_ratio = 0.1
    max_train_img_size = 736
    max_predict_img_size = 2400

    assert max_train_img_size in [256, 384, 512, 640, 736], 'max_train_img_size must in [256~736]'




def parse(self,kwargs):
    '''
    update the config params
    :param self:
    :param kwargs:
    :return:
    '''
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("Warning:opt has not attribute ^s" %k)

        setattr(self,k,v)

    print('use config:')
    for k,v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k,getattr(self,k))
    print("end the parse!!!")


DefaultConfig.parse=parse
opt=DefaultConfig()
