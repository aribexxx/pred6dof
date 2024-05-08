import logging
import os
from math import floor
import numpy as np
import pandas as pd
import toml
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from statsmodels.iolib.smpickle import save_pickle
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults, ar_select_order
from evaluator import Evaluator
from utils import get_csv_files

#TODO: kalman runner need 2 implementation, and one common interface, one is do prediction using local data, another using realtime data.
class KalmanRunner():
    """Runs the Kalman predictor over all traces"""
    def __init__(self, pred_window, dataset_path, results_path):
        config_path = os.path.join(os.getcwd(), 'config.toml')
        self.cfg = toml.load(config_path)
        self.dt = self.cfg['dt']
        self.pred_window = pred_window * 1e-3   # convert to seconds
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.coords = self.cfg['pos_coords'] + self.cfg['quat_coords']
        self.kf = KalmanFilter(dim_x = self.cfg['dim_x'], dim_z = self.cfg['dim_z'])
        setattr(self.kf, 'x_pred', self.kf.x)

        # First-order motion model: insert dt into the diagonal blocks of F
        f = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.kf.F = block_diag(f, f, f, f, f, f, f)

        # Inserts 1 into the blocks of H to select the measuremetns
        np.put(self.kf.H, np.arange(0, self.kf.H.size, self.kf.dim_x + 2), 1.0)
        self.kf.R *= self.cfg['var_R']
        Q_pos = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.cfg['var_Q_pos'], block_size=3)
        Q_ang = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.cfg['var_Q_ang'], block_size=4)
        self.kf.Q = block_diag(Q_pos, Q_ang)

    def reset(self):
        logging.debug("Reset Kalman filter")
        self.kf.x = np.zeros((self.cfg['dim_x'], 1))
        self.kf.P = np.eye(self.cfg['dim_x'])

    def lookahead(self):
        self.kf.x_pred = np.dot(self.kf.F_lookahead, self.kf.x)

    """Runs the Kalman predictor over all traces"""
    def __init__(self, pred_window, headpose, pred_results_path):
        config_path = os.path.join(os.getcwd(), 'config.toml')
        self.cfg = toml.load(config_path)
        self.dt = self.cfg['dt']
        self.pred_window = pred_window * 1e-3   # convert to seconds
        self.headpose = headpose
        self.results_path = pred_results_path
        self.coords = self.cfg['pos_coords'] + self.cfg['quat_coords']
        self.kf = KalmanFilter(dim_x = self.cfg['dim_x'], dim_z = self.cfg['dim_z'])
        setattr(self.kf, 'x_pred', self.kf.x)

        # First-order motion model: insert dt into the diagonal blocks of F
        f = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.kf.F = block_diag(f, f, f, f, f, f, f)

        # Inserts 1 into the blocks of H to select the measuremetns
        np.put(self.kf.H, np.arange(0, self.kf.H.size, self.kf.dim_x + 2), 1.0)
        self.kf.R *= self.cfg['var_R']
        Q_pos = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.cfg['var_Q_pos'], block_size=3)
        Q_ang = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.cfg['var_Q_ang'], block_size=4)
        self.kf.Q = block_diag(Q_pos, Q_ang)

    def reset(self):
        logging.debug("Reset Kalman filter")
        self.kf.x = np.zeros((self.cfg['dim_x'], 1))
        self.kf.P = np.eye(self.cfg['dim_x'])

    def lookahead(self):
        self.kf.x_pred = np.dot(self.kf.F_lookahead, self.kf.x)

    #read from path "./data/alvr.csv", then predict
    def run_all_with_csv_data(self, trace_path,w):

        # Adjust F depending on the lookahead time
        f_l = np.array([[1.0, w], [0.0, 1.0]])
        setattr(self.kf, 'F_lookahead', block_diag(f_l, f_l, f_l, f_l, f_l, f_l, f_l))
        # Read trace from CSV file
        df_trace = pd.read_csv("./data/alvr.csv")
        print(df_trace)
        xs, covs, x_preds = [], [], []
        zs = df_trace[self.coords].to_numpy()
        z_prev = np.zeros(7)
        for z in zs:
            # sign_array = -np.sign(z_prev[3:]) * np.sign(z[3:])
            # sign_flipped = all(e == 1 for e in sign_array)
            # if sign_flipped:
            #     logging.debug("A sign flip occurred.")
            #     self.reset()
            self.kf.predict()
            self.kf.update(z)
            self.lookahead()
            xs.append(self.kf.x)
            covs.append(self.kf.P)
            x_preds.append(self.kf.x_pred)
            print(f"origin:{z}, pred:{self.kf.x_pred}")
            z_prev = z
        
        # Compute evaluation metrics
        # xs = np.array(xs).squeeze()
        # covs = np.array(covs).squeeze()
        # x_preds = np.array(x_preds).squeeze()
        # pred_step = int(lat / self.dt)
        # eval = Evaluator(zs, x_preds[:, ::2], pred_step)
        # eval.eval_kalman()
        # metrics = np.array(list(eval.metrics.values()))
        # euc_dists = eval.euc_dists
        # ang_dists = np.rad2deg(eval.ang_dists)
        # print(f"metrics: {metrics}, euc_dists:{euc_dists}, ang_dists:{ang_dists}")
        # return metrics, euc_dists, ang_dists
    

    
    #TODO: directly take motion data from socket and predict
    #run one prediction windows settings 
    def run_with_single_pred_win(self,motions_array,w):
          # Adjust F depending on the lookahead time
        f_l = np.array([[1.0, 0.0], [0.0, 1.0]])
        setattr(self.kf, 'F_lookahead', block_diag(f_l, f_l, f_l, f_l, f_l, f_l, f_l))

        xs, covs, x_preds = [], [], []
        z_prev = np.zeros(7)
        current_measure_z = motions_array
        print(f"current messsaure:{ current_measure_z}")
        self.kf.predict()
        self.kf.update(current_measure_z)
        self.lookahead()
        xs.append(self.kf.x)
        covs.append(self.kf.P)
        x_preds.append(self.kf.x_pred)
        z_prev = current_measure_z
        print(f"origin:{current_measure_z}, pred:{self.kf.x_pred}")
        
        # Compute evaluation metrics
        # xs = np.array(xs).squeeze()
        # covs = np.array(covs).squeeze()
        # x_preds = np.array(x_preds).squeeze()
        # pred_step = int(w / self.dt)
        # eval = Evaluator(measurement_data, x_preds[:, ::2], pred_step)
        # eval.eval_kalman()
        # metrics = np.array(list(eval.metrics.values()))
        # euc_dists = eval.euc_dists
        # ang_dists = np.rad2deg(eval.ang_dists)       
        # return metrics, euc_dists, ang_dists
    #run all prediction windows settings in a loop
    
    # def run_with_many_predict_win(self, motion):            
    #     for w in self.pred_window:
    #             logging.info("Prediction window = %s ms", w * 1e3)
    #             self.reset()

    #             metrics, euc_dists, ang_dists = self.run_with_single_pred_win(w,motion)
    #             np.save(os.path.join(dists_path, 
    #                                     'euc_dists_{}_{}ms.npy'.format(basename, int(w*1e3))), euc_dists)
    #             np.save(os.path.join(dists_path, 
    #                                     'ang_dists_{}_{}ms.npy'.format(basename, int(w*1e3))), ang_dists)
    #             result_single = list(np.hstack((basename, w, metrics)))
    #             results.append(result_single)
    #             print("--------------------------------------------------------------")
    #     df_results = pd.DataFrame(results, columns=['Trace', 'LAT', 'mae_euc', 'mae_ang',
    #                                                 'rmse_euc', 'rmse_ang'])
    #     df_results.to_csv(os.path.join(self.results_path, 'res_kalman.csv'), index=False)
        
    #     return