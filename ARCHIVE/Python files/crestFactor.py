"""https://gist.github.com/endolith/2c786bf5b53b99ca3879"""


def ac_rms(signal):
    """Return the RMS level of the signal after removing any fixed DC offset"""
    return rms_flat(signal - mean(signal))

 def properties(signal, samplerate):
    """Return a list of some wave properties for a given 1-D signal"""
    signal_level = ac_rms(signal)
    peak_level = max(max(signal.flat),-min(signal.flat))
    crest_factor = peak_level/signal_level
   	'Crest factor: %.3f (%.3f dB)' % (crest_factor, dB(crest_factor))

 def dB(level):
     """Return a level in decibels.
     
     Decibels are relative to the RMS level of a full-scale square wave 
     of peak amplitude 1.0 (dBFS).
     
     A full-scale square wave is 0 dB
     A full-scale sine wave is -3.01 dB
     """
     return 20 * log10(level)