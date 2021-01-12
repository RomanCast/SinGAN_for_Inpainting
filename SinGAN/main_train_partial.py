from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training_partialconv import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Inpainting')
    parser.add_argument('--ref_name', help='reference image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--structured', action='store_true')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    masks =[]
    NoiseAmp = []
    mask_dir = '%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:])
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, masks, mask_dir, NoiseAmp)
        Gs, Zs, reals, masks, NoiseAmp = functions.load_trained_pyramid_withMasks(opt)
        SinGAN_generate_partial(Gs,Zs,reals,masks,NoiseAmp,opt)
