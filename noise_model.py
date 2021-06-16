def build_noise(**extras):
    from prospect.likelihood.kernels import PhotoCal, Uncorrelated
    from prospect.likelihood.noise_model import NoiseModel

    # This is the correlated noise kernel.  It's basically just a J x J
    # covariance matrix with zeros except for elements corrsponding to a pair of
    # filters that are both in 'nir_bandnames'.  Since calibration is
    # multiplicative, it will get outer multiplied by 'phot' via the weight_by keyword
    kcorr = PhotoCal(parnames=["nir_offset", "nir_bandnames"])
    # Here's the independent noise.  This is just the indentity matrix, scaled
    # by 'phot_jitter'. It will be outer multiplied b 'phot_unc'
    kind = Uncorrelated(parnames=["phot_jitter"])

    phot_noise = NoiseModel(metric_name="filternames", mask_name="phot_mask",
                            kernels=[kind, kcorr], weight_by=['phot', 'phot_unc'])
    return None, phot_noise

