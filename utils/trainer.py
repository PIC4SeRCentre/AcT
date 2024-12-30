# GENERAL LIBRARIES 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import math
import numpy as np
import joblib
from pathlib import Path
# MACHINE LEARNING LIBRARIES
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# OPTUNA
import optuna
from optuna.trial import TrialState
# CUSTOM LIBRARIES
from utils.transformer import TransformerEncoder, PatchClassEmbedding, Patches
from utils.data import load_mpose, load_kinetics, random_flip, random_noise, one_hot
from utils.tools import CustomSchedule, CosineSchedule
from utils.tools import Logger


# TRAINER CLASS 
class Trainer:
    def __init__(self, config, logger, split=1, fold=0):
        self.config = config
        self.logger = logger
        self.split = split
        self.fold = fold
        self.trial = None
        self.bin_path = self.config['MODEL_DIR']
        
        self.model_size = self.config['MODEL_SIZE']
        self.n_heads = self.config[self.model_size]['N_HEADS']
        self.n_layers = self.config[self.model_size]['N_LAYERS']
        self.embed_dim = self.config[self.model_size]['EMBED_DIM']
        self.dropout = self.config[self.model_size]['DROPOUT']
        self.mlp_head_size = self.config[self.model_size]['MLP']
        self.activation = tf.nn.gelu
        self.d_model = 64 * self.n_heads
        self.d_ff = self.d_model * 4
        self.pos_emb = self.config['POS_EMB']

        
    def build_act(self, transformer):
        inputs = tf.keras.layers.Input(shape=(self.config[self.config['DATASET']]['FRAMES'] // self.config['SUBSAMPLE'], 
                                              self.config[self.config['DATASET']]['KEYPOINTS'] * self.config['CHANNELS']))
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        x = PatchClassEmbedding(self.d_model, self.config[self.config['DATASET']]['FRAMES'] // self.config['SUBSAMPLE'], 
                                pos_emb=None)(x)
        x = transformer(x)
        x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
        x = tf.keras.layers.Dense(self.mlp_head_size)(x)
        outputs = tf.keras.layers.Dense(self.config[self.config['DATASET']]['CLASSES'])(x)
        return tf.keras.models.Model(inputs, outputs)

    
    def get_model(self):
        transformer = TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.n_layers)
        self.model = self.build_act(transformer)
        
        self.train_steps = np.ceil(float(self.train_len)/self.config['BATCH_SIZE'])
        self.test_steps = np.ceil(float(self.test_len)/self.config['BATCH_SIZE'])
        
        if self.config['SCHEDULER']:
            lr = CustomSchedule(self.d_model, 
                                warmup_steps=self.train_steps*self.config['N_EPOCHS']*self.config['WARMUP_PERC'],
                                decay_step=self.train_steps*self.config['N_EPOCHS']*self.config['STEP_PERC'])

        else:
            lr = 3 * 10**self.config['LR_MULT']
        
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=self.config['WEIGHT_DECAY'])

        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])

        self.name_model_bin = f"{self.config['MODEL_NAME']}_{self.config['MODEL_SIZE']}_{self.split}_{self.fold}.h5"

        self.checkpointer = tf.keras.callbacks.ModelCheckpoint(self.bin_path + self.name_model_bin,
                                                               monitor="val_accuracy",
                                                               save_best_only=True,
                                                               save_weights_only=True)
        return
    
    
    def get_data(self):
        if self.config['DATASET'] == 'kinetics':
            train_gen, val_gen, test_gen, self.train_len, self.test_len = load_kinetics(self.config, self.fold)
            
            self.ds_train = tf.data.Dataset.from_generator(train_gen, 
                                                           output_types=('float32', 'uint8'))
            self.ds_val = tf.data.Dataset.from_generator(val_gen, output_types=('float32', 'uint8'))
            self.ds_test = tf.data.Dataset.from_generator(test_gen, output_types=('float32', 'uint8'))
            
        else:
            X_train, y_train, X_test, y_test = load_mpose(self.config['DATASET'], self.split, 
                                                          legacy=self.config['LEGACY'], verbose=False)
            self.train_len = len(y_train)
            self.test_len = len(y_test)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                              test_size=self.config['VAL_SIZE'],
                                                              random_state=self.config['SEEDS'][self.fold],
                                                              stratify=y_train)
                
            self.ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            self.ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            self.ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            
        self.ds_train = self.ds_train.map(lambda x,y : one_hot(x,y,self.config[self.config['DATASET']]['CLASSES']), 
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.cache()
        self.ds_train = self.ds_train.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.map(random_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.shuffle(X_train.shape[0])
        self.ds_train = self.ds_train.batch(self.config['BATCH_SIZE'])
        self.ds_train = self.ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        
        self.ds_val = self.ds_val.map(lambda x,y : one_hot(x,y,self.config[self.config['DATASET']]['CLASSES']), 
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_val = self.ds_val.cache()
        self.ds_val = self.ds_val.batch(self.config['BATCH_SIZE'])
        self.ds_val = self.ds_val.prefetch(tf.data.experimental.AUTOTUNE)

        
        self.ds_test = self.ds_test.map(lambda x,y : one_hot(x,y,self.config[self.config['DATASET']]['CLASSES']), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_test = self.ds_test.cache()
        self.ds_test = self.ds_test.batch(self.config['BATCH_SIZE'])
        self.ds_test = self.ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        
    def get_random_hp(self):
        # self.config['RN_STD'] = self.trial.suggest_float("RN_STD", 0, 0.03, 0.01)
        self.config['WEIGHT_DECAY'] = self.trial.suggest_float("WD", 1e-5, 1e-3, step=1e-5)
        self.config['N_EPOCHS'] = int(self.trial.suggest_float("EPOCHS", 50, 100, step=10))
        self.config['WARMUP_PERC'] = self.trial.suggest_float("WARMUP_PERC", 0.1, 0.4, step=0.1)
        self.config['LR_MULT'] = self.trial.suggest_float("LR_MULT", -5, -4, step=1)
        self.config['SCHEDULER'] = self.trial.suggest_categorical("SCHEDULER", [False, False])
        
        # self.logger.save_log('\nRN_STD: {:.2e}'.format(self.config['RN_STD']))
        self.logger.save_log('\nEPOCHS: {}'.format(self.config['N_EPOCHS']))
        self.logger.save_log('WARMUP_PERC: {:.2e}'.format(self.config['WARMUP_PERC']))
        self.logger.save_log('WEIGHT_DECAY: {:.2e}'.format(self.config['WEIGHT_DECAY']))
        self.logger.save_log('LR_MULT: {:.2e}'.format(self.config['LR_MULT']))
        self.logger.save_log('SCHEDULER: {}\n'.format(self.config['SCHEDULER']))
        
    def do_training(self):
        self.get_data()
        self.get_model()

        self.model.fit(self.ds_train,
                       epochs=self.config['N_EPOCHS'], initial_epoch=0,
                       validation_data=self.ds_val,
                       callbacks=[self.checkpointer], verbose=self.config['VERBOSE'],
                       #steps_per_epoch=int(self.train_steps*0.9),
                       #validation_steps=self.train_steps//9
                      )
        accuracy_test, balanced_accuracy = self.evaluate(weights=self.bin_path+self.name_model_bin)      
        return accuracy_test, balanced_accuracy

    def evaluate(self, weights=None):
        if weights is not None:
            self.model.load_weights(self.bin_path+self.name_model_bin)  
        else:
            self.model.load_weights(self.config['WEIGHTS'])

        _, accuracy_test = self.model.evaluate(self.ds_test, steps=self.test_steps)
        if self.config['DATASET'] == 'kinetics':
            g = list(self.ds_test.take(self.test_steps).as_numpy_iterator())
            X = [e[0] for e in g]
            X = np.vstack(X)
            y = [e[1] for e in g]
            y = np.vstack(y)
        else:
            X, y = tuple(zip(*self.ds_test))
        
        y_pred = np.argmax(tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1),axis=1)
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(tf.math.argmax(tf.concat(y, axis=0), axis=1), y_pred)

        text = f"Accuracy Test: {accuracy_test} <> Balanced Accuracy: {balanced_accuracy}\n"
        self.logger.save_log(text)
        
        return accuracy_test, balanced_accuracy

    def do_test(self):
        self.get_data()
        self.get_model()
        accuracy_test, balanced_accuracy = self.evaluate()
        return accuracy_test, balanced_accuracy
    
    def objective(self, trial):
        self.trial = trial     
        self.get_random_hp()
        _, bal_acc = self.do_training()
        return bal_acc
        
    def do_benchmark(self):
        for split in range(1, self.config['SPLITS']+1):      
            self.logger.save_log(f"----- Start Split {split} ----\n")
            self.split = split
            
            acc_list = []
            bal_acc_list = []

            for fold in range(self.config['FOLDS']):
                self.logger.save_log(f"- Fold {fold+1}")
                self.fold = fold
                
                acc, bal_acc = self.do_training()

                acc_list.append(acc)
                bal_acc_list.append(bal_acc)
                
            np.save(self.config['RESULTS_DIR'] + self.config['MODEL_NAME'] + '_' + self.config['DATASET'] + f'_{split}_accuracy.npy', acc_list)
            np.save(self.config['RESULTS_DIR'] + self.config['MODEL_NAME'] + '_' + self.config['DATASET'] + f'_{split}_balanced_accuracy.npy', bal_acc_list)

            self.logger.save_log(f"---- Split {split} ----")
            self.logger.save_log(f"Accuracy mean: {np.mean(acc_list)}")
            self.logger.save_log(f"Accuracy std: {np.std(acc_list)}")
            self.logger.save_log(f"Balanced Accuracy mean: {np.mean(bal_acc_list)}")
            self.logger.save_log(f"Balanced Accuracy std: {np.std(bal_acc_list)}")
        
    def do_random_search(self):
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(study_name='{}_random_search'.format(self.config['MODEL_NAME']),
                                         direction="maximize", pruner=pruner)
        self.study.optimize(lambda trial: self.objective(trial),
                            n_trials=self.config['N_TRIALS'])

        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        self.logger.save_log("Study statistics: ")
        self.logger.save_log(f"  Number of finished trials: {len(self.study.trials)}")
        self.logger.save_log(f"  Number of pruned trials: {len(pruned_trials)}")
        self.logger.save_log(f"  Number of complete trials: {len(complete_trials)}")

        self.logger.save_log("Best trial:")

        self.logger.save_log(f"  Value: {self.study.best_trial.value}")

        self.logger.save_log("  Params: ")
        for key, value in self.study.best_trial.params.items():
            self.logger.save_log(f"    {key}: {value}")

        joblib.dump(self.study,
          f"{self.config['RESULTS_DIR']}/{self.config['MODEL_NAME']}_{self.config['DATASET']}_random_search_{str(self.study.best_trial.value)}.pkl")
        
    def return_model(self):
        return self.model
