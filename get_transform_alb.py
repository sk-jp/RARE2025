import albumentations as alb
import albumentations.pytorch as albp


def get_transform(conf_augmentation, replay=False):
    """ Get augmentation function
        Args:
            conf_augmentation (Dict): dictionary of augmentation parameters
            replay (bool): if True, return ReplayCompose object. if False, return Compose object.
    """
    def get_object(trans):
        if trans.name in {'Compose', 'OneOf'}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(alb, trans.name)(augs_tmp, **trans.params)

        if hasattr(alb, trans.name):
            return getattr(alb, trans.name)(**trans.params)
        else:
            return eval(trans.name)(**trans.params)

    if conf_augmentation is None:
        augs = list()
    else:
        augs = [get_object(aug) for aug in conf_augmentation]

    augs.extend([
        albp.ToTensorV2()
    ])

#    print('augs:', augs)

    if replay:
        return alb.ReplayCompose(augs)
    else:
        return alb.Compose(augs)
