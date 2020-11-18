from __future__ import division
from builtins import str
from builtins import range

from sciunit import Score
import numpy
from sciunit.utils import assert_dimensionless
import collections

class ZScore_backpropagatingAP_OLM(Score):
    """
    Average of Z scores. A float indicating the average of standardized difference
    from reference means for back-propagating AP amplitudes.
    """

    def __init__(self, score, related_data={}):

        if not isinstance(score, Exception) and not isinstance(score, float):
            raise InvalidScoreError("Score must be a float.")
        else:
            super(ZScore_backpropagatingAP_OLM,self).__init__(score, related_data=related_data)

    @classmethod
    def compute(cls, observation, prediction, distances):
        """Computes average of z-scores from observation and prediction for back-propagating AP amplitudes"""
        errors = {'active_Na':{}, 'blocked_Na':{}}
        
        for i in range (0, len(distances)):
            
            p_value_active = prediction['active_Na']['model_AP1_amp_dendpersoma_at_'+str(distances[i])+'um']['mean']
            p_value_blocked = prediction['blocked_Na']['model_AP1_amp_dendpersoma_at_'+str(distances[i])+'um']['mean']
            o_mean_active = observation['active_Na']['mean_AP1_amp_dendpersoma_at_'+str(distances[i])+'um']
            o_mean_blocked = observation['blocked_Na']['mean_AP1_amp_dendpersoma_at_'+str(distances[i])+'um']
            o_std_active = observation['active_Na']['std_AP1_amp_dendpersoma_at_'+str(distances[i])+'um']
            o_std_blocked = observation['blocked_Na']['std_AP1_amp_dendpersoma_at_'+str(distances[i])+'um']
            
            try:
                error_active = abs(p_value_active - o_mean_active)/o_std_active
                error_active = assert_dimensionless(error_active)
            except (TypeError,AssertionError) as e:
                error_active = e
            errors['active_Na'] = {'AP1_amp_at_'+str(distances[i]) : error_active} 
            
            try:
                error_blocked = abs(p_value_blocked - o_mean_blocked)/o_std_blocked
                error_blocked = assert_dimensionless(error_blocked)
            except (TypeError,AssertionError) as e:
                error_blocked = e
            errors['blocked_Na'] = {'AP1_amp_at_'+str(distances[i]) : error_blocked}


        score_avg = numpy.nanmean(numpy.array([error_active, error_blocked]))

        return score_avg, errors

    def __str__(self):

        return 'Z_score_avg = %.2f' % self.score


