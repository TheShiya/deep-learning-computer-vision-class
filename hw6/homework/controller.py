import pystk


target_velocity = 24

def control(aim_point, current_vel):
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """ 
    #print(aim_point)

    steer_factor = 2
    steer_x_threshold = 0.05
    drift_x_threshold = 3
    drift_vel_threshold = 18
    brake_x_threshold = 6
    brake_vel_threshold = 15

    x, y, z = aim_point

    if current_vel < target_velocity:
        action.acceleration = 1
    if abs(x) >= steer_x_threshold:
    	action.steer = steer_factor * x
    if abs(x) >= drift_x_threshold: 	
    	action.acceleration = 1 - current_vel/target_velocity
    	if current_vel > drift_vel_threshold:
    		action.drift = True
    if abs(x) >= brake_x_threshold:
    	action.acceleration = 0.05
    	if current_vel > brake_vel_threshold:
    		action.brake = True
    		action.drift = True

    return action

'''
python -m homework.controller zengarden lighthouse hacienda snowtuxpeak cornfield_crossing scotland
'''


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
