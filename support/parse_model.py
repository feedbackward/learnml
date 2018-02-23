
# parse models.

import support.models as md

def model(dinfo):
    '''
    A general-purpose wrapper for model classes.

    Input: a data info object.

    Output: an instance of the desired model.
    '''
    
    # Return the appropriate model object.
    if dinfo.mname == "LinReg":
        return md.LinReg(dinfo)

    if dinfo.mname == "Encoder":
        return md.Encoder(dinfo)

    if dinfo.mname == "NoisyOpt":
        return md.NoisyOpt(dinfo)


