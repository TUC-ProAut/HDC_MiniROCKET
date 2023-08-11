# Kenny Schlegel, Peer Neubert, Peter Protzel
#
# HDC-MiniROCKET : Explicit Time Encoding in Time Series Classification with Hyperdimensional Computing
# Copyright (C) 2022 Chair of Automation Technology / TU Chemnitz

from data.dataset_utils import *
import logging
from models.HDC_MINIROCKET import HDC_MINIROCKET_model
from models.MINIROCKET import MINIROCKET_model
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, confusion_matrix

# self.config logger
logger = logging.getLogger('log')

class NetTrial():
    '''
    high level class for training and evaluating 
    '''
    
    def __init__(self,args):           
        # set self.config parameter
        exec('self.config = ' + args.config + '()')
        self.config.HDC_dim = args.HDC_dim
        self.config.scale = args.scale
        self.config.dataset = args.dataset
        self.config.model_name = args.model
        self.config.stat_iterations = args.stat_iterations
        self.config.ensemble_idx = args.ensemble_idx
        self.config.normalize_data = args.normalize
        self.config.dataset_path = args.dataset_path

        # load data at initialization
        self.load_data()

    def load_data(self):
        # load dataset initially
        self.data = load_dataset(self.config.dataset, self.config)
        X_train = self.data[0]
        X_test = self.data[1]
        y_train = self.data[2]
        y_test = self.data[3]

        self.config.n_classes = len(np.unique(y_train))
        self.config.n_inputs = X_train.shape[1]
        self.config.n_steps = X_train.shape[2]

        # if train test data not a list, create one (k-fold data set loading returns a list of splits - therefore we
        # are handling all training set as lists, even if they only contain one set)
        if type(X_train) == list:
            print("given data is not a list")
            self.X_train_list = X_train
            self.X_test_list = X_test
            self.y_train_list = y_train
            self.y_test_list = y_test
        else:
            self.X_train_list = [X_train]
            self.X_test_list = [X_test]
            self.y_train_list = [y_train]
            self.y_test_list = [y_test]

        if self.config.model_name == "HDC_MINIROCKET":
            self.model = HDC_MINIROCKET_model(self.config)
        elif self.config.model_name == "MINIROCKET":
            self.model = MINIROCKET_model(self.config)

    def train(self):
        '''
        training procedure
        @return:
        '''

        acc_stat = []
        f1_stat = []

        # statistical iteration
        for stat_it in range(self.config.stat_iterations):
            logger.info('Statistial iteration: ' + str(stat_it))
    
            # train for each element in list (that is why we need list form, even if it contains only one element)
            logger.info('Training data contains ' + str(len(self.X_train_list)) + ' training instances...')
            f1s = []
            accs = []

            for it in range(len(self.X_train_list)):
                logger.info(('.......'))
                logger.info('instance ' + str(it) + ':')

                X_train = self.X_train_list[it]
                X_test = self.X_test_list[it]
                y_train = self.y_train_list[it]
                y_test = self.y_test_list[it]

                # self.config.HDC_dim = X_train.shape[1]
                logger.info('Training dataset shape: ' + str(X_train.shape) + str(y_train.shape))
                logger.info('Test dataset shape: ' + str(X_test.shape) + str(y_test.shape))

                self.config.train_count = len(X_train)
                self.config.test_data_count = len(X_test)
                self.model.config = self.config

                # train the model
                self.model.train_model(X_train,y_train,X_test,y_test)

                # evaluate the model
                y_pred, y_scores = self.model.eval_model(X_train, y_train, X_test, y_test)
                # evaluate the results
                logger.info('Results on test data: ')
                report = classification_report(y_test.astype(int), y_pred, output_dict=True)

                # accuracy and f1 score
                acc = report['accuracy']
                f1 = f1_score(y_test.astype(int), y_pred, average='weighted')

                logger.info('Accuracy: ' + str(acc))
                logger.info('F1 score: ' + str(f1))

                accs.append(acc)
                f1s.append(f1)

                acc_stat.append(np.mean(accs))
                f1_stat.append((np.mean(f1s)))

                idx = self.config.ensemble_idx

                try:
                    file_path = 'results/' + self.config.model_name + '_' + self.config.dataset + '_' + \
                           self.config.note

                    # write other results to excel
                    file = file_path + '_results.xlsx'
                    file_f1 = file_path + '_results_f1.xlsx'
                    file_time = file_path + '_time.xlsx'
                    acc_df = pd.DataFrame({'data': acc}, index=[0])
                    f1_df = pd.DataFrame({'data': f1}, index=[0])
                    idx_df = pd.DataFrame({'data': idx}, index=[0])
                    # write index
                    append_df_to_excel(file, pd.DataFrame({self.config.dataset + '_idx': []}),
                                       startcol=0, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_f1, pd.DataFrame({self.config.dataset + '_idx': []}),
                                       startcol=0, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({self.config.dataset + '_idx': []}),
                                       startcol=0, index=False, header=True,
                                       startrow=0)

                    if self.config.best_scale:
                        header_name = pd.DataFrame({'acc_at_best_scale' :[],
                                                    'best_scale':[]})
                    else:
                        header_name = pd.DataFrame({'scale_idx' + str(self.config.scale_idx): []})
                    # files for the normal results
                    append_df_to_excel(file, header_name,
                                       startcol=self.config.scale_idx + 1, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file, acc_df,
                                       index=False, header=False, startrow=idx + 1,
                                       startcol=self.config.scale_idx + 1)

                    if self.config.best_scale:
                        append_df_to_excel(file, pd.DataFrame({'best_scale': str(self.config.scale)}, index=[0]),
                                           index=False, header=False, startrow=idx + 1,
                                           startcol=2)

                    # files for the f1 results
                    append_df_to_excel(file_f1, header_name,
                                        startcol=self.config.scale_idx + 1, index=False, header=True,
                                        startrow=0)
                    append_df_to_excel(file_f1, f1_df,
                                        index=False, header=False, startrow=idx + 1,
                                        startcol=self.config.scale_idx + 1)

                    append_df_to_excel(file, idx_df,
                                       startcol=0, index=False, header=False,
                                       startrow=idx + 1)
                    append_df_to_excel(file_f1, idx_df,
                                        startcol=0, index=False, header=False,
                                        startrow=idx + 1)

                    # write run-time results to file
                    append_df_to_excel(file_time, idx_df,
                                       startcol=0, index=False, header=False,
                                       startrow=idx + 1)
                    append_df_to_excel(file_time, pd.DataFrame({'prep_time_train': []}),
                                       startcol=1, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({'data': self.model.train_preproc}, index=[0]),
                                       startcol=1, index=False, header=False,
                                       startrow=idx + 1)
                    append_df_to_excel(file_time, pd.DataFrame({'prep_time_test': []}),
                                       startcol=2, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({'data': self.model.test_preproc}, index=[0]),
                                       startcol=2, index=False, header=False,
                                       startrow=idx + 1)
                    append_df_to_excel(file_time, pd.DataFrame({'train_time': []}),
                                       startcol=3, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({'data': self.model.training_time}, index=[0]),
                                       startcol=3, index=False, header=False,
                                       startrow=idx + 1)
                    append_df_to_excel(file_time, pd.DataFrame({'inf_time': []}),
                                       startcol=4, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({'data': self.model.testing_time}, index=[0]),
                                       startcol=4, index=False, header=False,
                                       startrow=idx + 1)

                except Exception as e:
                    logger.info(e)


    def eval(self):
        '''
        evaluating procedure
        @return:
        '''

        f1s = []
        accs = []
        for it in range(len(self.X_test_list)):
            logger.info(('.......'))
            logger.info('instance ' + str(it) + ':')

            X_train = self.X_train_list[it]
            X_test = self.X_test_list[it]
            y_train = self.y_train_list[it]
            y_test = self.y_test_list[it]

            logger.info('Test dataset shape: ' + str(X_test.shape) + str(y_test.shape))

            self.config.test_data_count = len(X_test)
            self.model.config = self.config

            # evaluate the model
            acc, f1, confusion_matrix = self.model.eval_model(X_train,y_train,X_test,y_test,fold=it)

            accs.append(acc)
            f1s.append(f1)

        with open('results/computing_time_inference_' + self.config.dataset + '_' + self.config.model_name + '.txt',
                  'a') as file:
            file.write(str(self.config.ensemble_idx) + '\t' +
                       str(self.model.test_preproc) + '\t'
                       + str(self.model.testing_time) + '\n'
                       )

        logger.info('Accuracy results: ' + str(np.mean(accs)))
        logger.info('F1 scores: ' + str(np.mean(f1s)))



def append_df_to_excel(filename, df, sheet_name='Sheet1', startrow=None,
                       truncate_sheet=False,
                       **to_excel_kwargs):
    """
    code from https://gist.github.com/fredpedroso/590e54d4f07d0ae2d20d0ec0b190d5ff

    Append a DataFrame [df] to existing Excel file [filename]
    into [sheet_name] Sheet.
    If [filename] doesn't exist, then this function will create it.
    Parameters:
      filename : File path or existing ExcelWriter
                 (Example: '/path/to/file.xlsx')
      df : dataframe to save to workbook
      sheet_name : Name of sheet which will contain DataFrame.
                   (default: 'Sheet1')
      startrow : upper left cell row to dump data frame.
                 Per default (startrow=None) calculate the last row
                 in the existing DF and write to the next row...
      truncate_sheet : truncate (remove and recreate) [sheet_name]
                       before writing DataFrame to Excel file
      to_excel_kwargs : arguments which will be passed to `DataFrame.to_excel()`
                        [can be dictionary]
    Returns: None
    """
    # ignore [engine] parameter if it was passed
    if 'engine' in to_excel_kwargs:
        to_excel_kwargs.pop('engine')

    if os.path.isfile(filename):
        writer = pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists="overlay")
    else:
        # check if directory exists
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        writer = pd.ExcelWriter(filename, mode='w', engine='openpyxl')

    # write out the new sheet
    df.to_excel(writer, sheet_name, startrow=startrow, **to_excel_kwargs)

    # save the workbook
    writer.close()