from anycam.models.anycam import AnyCam


def make_depth_predictor(conf, **kwargs):
    enc_type = conf["type"]
    if enc_type == "depthanything":
        from anycam.models.depth_predictor_wrapper import DepthAnythingWrapper
        predictor = DepthAnythingWrapper.from_conf(conf, **kwargs)
    elif enc_type == "unidepth":
        from anycam.models.depth_predictor_wrapper import UniDepthV2Wrapper
        predictor = UniDepthV2Wrapper.from_conf(conf, **kwargs)
    elif enc_type == "metric3d":
        from anycam.models.depth_predictor_wrapper import Metric3DV2Wrapper
        predictor = Metric3DV2Wrapper.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported depth predictor type: {enc_type}")
    return predictor


def make_pose_predictor(conf, **kwargs):
    enc_type = conf["type"]
    if enc_type == "anycam":
        # ===== DA3 INTEGRATION: Pass use_da3_calibration through kwargs =====
        use_da3_calibration = kwargs.get("use_da3_calibration", False)
        predictor = AnyCam(conf, use_da3_calibration=use_da3_calibration)
        # ===== DA3 INTEGRATION: END =====
    else:
        raise NotImplementedError(f"Unsupported pose predictor type: {enc_type}")
    return predictor


def make_depth_aligner(conf, **kwargs):
    da_type = conf["type"]
    if da_type == "identity":
        aligner = None
    else:
        raise NotImplementedError(f"Unsupported depth aligner type: {da_type}")
    return aligner
