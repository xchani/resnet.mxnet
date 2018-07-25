import mxnet as mx

def spatial_pyramid_pooling_unit(data, num_filter, pool_scale, name, bn_mom=0.9, workspace=512):
    spp_pool = mx.symbol.Pooling(data=data, kernel=(pool_scale, pool_scale), stride=(pool_scale, pool_scale),
                                 pool_type='avg', name='spp_%s' % name)
    spp_pool_conv = mx.sym.Convolution(data=spp_pool, num_filter=num_filter, kernel=(1, 1), no_bias=True,
                                       workspace=workspace, name='spp_%s_conv' % name)
    spp_pool_bn = mx.sym.BatchNorm(data=spp_pool_conv, fix_gamma=False, momentum=bn_mom, eps=2e-5,
                                    name='spp_%s_bn' % name)
    spp_pool_relu = mx.sym.Activation(data=spp_pool_bn, act_type='relu', name='spp_%s_relu' % name)
    spp_pool_iterp = mx.symbol.UpSampling(spp_pool_relu, scale=pool_scale, sample_type='nearest',
                                          num_filter=num_filter, workspace=2048, name='spp_%s_iterp' % name)
    return spp_pool_iterp

def spatial_pyramid_pooling(data, num_filter, input_size, pool_list, bn_mom=0.9):
    for i in range(len(pool_list)):
        assert(input_size % pool_list[i] == 0)
    pool1 = spatial_pyramid_pooling_unit(data, num_filter, input_size/pool_list[0], 'pool1')
    pool2 = spatial_pyramid_pooling_unit(data, num_filter, input_size/pool_list[1], 'pool2')
    pool3 = spatial_pyramid_pooling_unit(data, num_filter, input_size/pool_list[2], 'pool3')
    pool4 = spatial_pyramid_pooling_unit(data, num_filter, input_size/pool_list[3], 'pool4')
    spp_concat = mx.sym.concat(data, pool1, pool2, pool3, pool4, name='spp_concat')
    spp_concat_conv = mx.sym.Convolution(data=spp_concat, num_filter=num_filter, kernel=(3, 3), stride=(1, 1),
                                         pad=(1, 1), workspace=4096, no_bias=True, name='spp_concat_conv')
    spp_concat_bn = mx.sym.BatchNorm(data=spp_concat_conv, fix_gamma=False, momentum=bn_mom, eps=2e-5,
                                     name='spp_concat_bn')
    spp_concat_relu = mx.sym.Activation(data=spp_concat_bn, act_type='relu', name='spp_concat_relu')
    return spp_concat_relu

