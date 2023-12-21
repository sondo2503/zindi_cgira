import timm


MODELS = {'convf_384': 'convformer_b36.sail_in1k_384',
          'convf_224': 'convformer_b36.sail_in1k'}


def get_model(config, num_class=5, pretrained=True):
    model = timm.create_model(model_name=config.model.name, pretrained=pretrained, num_classes=num_class)
    return model


# if __name__ == '__main__':
#     print('main')
#     model = get_model()
