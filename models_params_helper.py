from ast import literal_eval

def cnn_params(params, dict_cnn):
    """
    Reads the dictionary of the detection and classification CNN
    and puts the parameters in the corresponding fields of the params object.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    dict_cnn : dict
        The keys are the parameter names and they are associated to their values.
    """

    # CNN params
    params.batchsize = literal_eval(dict_cnn['batchsize'])
    params.nb_conv_layers = literal_eval(dict_cnn['nb_conv_layers']) # nb_conv+1 in total because input layer counted separately
    params.nb_dense_layers = literal_eval(dict_cnn['nb_dense_layers']) # nb_dense+1 in total because last layer is not associated to a dropout layer and different nb of nodes
    params.nb_filters = literal_eval(dict_cnn['nb_filters'])
    params.net_type = dict_cnn['net_type']
    params.filter_size = literal_eval(dict_cnn['filter_size'])
    params.pool_size = literal_eval(dict_cnn['pool_size'])
    params.nb_dense_nodes = literal_eval(dict_cnn['nb_dense_nodes'])
    params.dropout_proba = literal_eval(dict_cnn['dropout_proba'])
    # Adam
    params.learn_rate_adam = literal_eval(dict_cnn['learn_rate_adam'])
    params.beta_1 = literal_eval(dict_cnn['beta_1'])
    params.beta_2 = literal_eval(dict_cnn['beta_2'])
    params.epsilon = literal_eval(dict_cnn['epsilon'])
    # early stopping
    params.min_delta = literal_eval(dict_cnn['min_delta'])
    params.patience = literal_eval(dict_cnn['patience'])
    
def cnn_params_1(params, dict_cnn):
    """
    Reads the dictionary of the detection CNN
    and puts the parameters in the corresponding fields of the params object.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    dict_cnn : dict
        The keys are the parameter names and they are associated to their values.
    """

    # CNN params
    params.batchsize_1 = literal_eval(dict_cnn['batchsize'])
    params.nb_conv_layers_1 = literal_eval(dict_cnn['nb_conv_layers']) # nb_conv+1 in total because input layer counted separately
    params.nb_dense_layers_1 = literal_eval(dict_cnn['nb_dense_layers']) # nb_dense+1 in total because last layer is not associated to a dropout layer and different nb of nodes
    params.nb_filters_1 = literal_eval(dict_cnn['nb_filters'])
    params.net_type_1 = dict_cnn['net_type']
    params.filter_size_1 = literal_eval(dict_cnn['filter_size'])
    params.pool_size_1 = literal_eval(dict_cnn['pool_size'])
    params.nb_dense_nodes_1 = literal_eval(dict_cnn['nb_dense_nodes'])
    params.dropout_proba_1 = literal_eval(dict_cnn['dropout_proba'])
    # Adam
    params.learn_rate_adam_1 = literal_eval(dict_cnn['learn_rate_adam'])
    params.beta_1_1 = literal_eval(dict_cnn['beta_1'])
    params.beta_2_1 = literal_eval(dict_cnn['beta_2'])
    params.epsilon_1 = literal_eval(dict_cnn['epsilon'])
    # early stopping
    params.min_delta_1 = literal_eval(dict_cnn['min_delta'])
    params.patience_1 = literal_eval(dict_cnn['patience'])

def cnn_params_2(params, dict_cnn):
    """
    Reads the dictionary of the classification CNN
    and puts the parameters in the corresponding fields of the params object.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    dict_cnn : dict
        The keys are the parameter names and they are associated to their values.
    """

    # CNN params
    params.batchsize_2 = literal_eval(dict_cnn['batchsize'])
    params.nb_conv_layers_2 = literal_eval(dict_cnn['nb_conv_layers']) # nb_conv+1 in total because input layer counted separately
    params.nb_dense_layers_2 = literal_eval(dict_cnn['nb_dense_layers']) # nb_dense+1 in total because last layer is not associated to a dropout layer and different nb of nodes
    params.nb_filters_2 = literal_eval(dict_cnn['nb_filters'])
    params.net_type_2 = dict_cnn['net_type']
    params.filter_size_2 = literal_eval(dict_cnn['filter_size'])
    params.pool_size_2 = literal_eval(dict_cnn['pool_size'])
    params.nb_dense_nodes_2 = literal_eval(dict_cnn['nb_dense_nodes'])
    params.dropout_proba_2 = literal_eval(dict_cnn['dropout_proba'])
    # Adam
    params.learn_rate_adam_2 = literal_eval(dict_cnn['learn_rate_adam'])
    params.beta_1_2 = literal_eval(dict_cnn['beta_1'])
    params.beta_2_2 = literal_eval(dict_cnn['beta_2'])
    params.epsilon_2 = literal_eval(dict_cnn['epsilon'])
    # early stopping
    params.min_delta_2 = literal_eval(dict_cnn['min_delta'])
    params.patience_2 = literal_eval(dict_cnn['patience'])
  
def xgboost_params(params, dict_xgboost):
    """
    Read the dictionary of the XGBoost
    and puts the parameters in the corresponding fields of the params object.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    dict_xgboost : dict
        The keys are the parameter names and they are associated to their values.
    """

    params.max_depth = literal_eval(dict_xgboost['max_depth'])
    params.eta = literal_eval(dict_xgboost['eta'])
    params.min_child_weight = literal_eval(dict_xgboost['min_child_weight'])
    params.n_estimators = literal_eval(dict_xgboost['n_estimators'])
    params.gamma_xgb = literal_eval(dict_xgboost['gamma_xgb'])
    params.subsample = literal_eval(dict_xgboost['subsample'])
    params.scale_pos_weight = literal_eval(dict_xgboost['scale_pos_weight'])

def resnet_params(params, dict_resnet):
    """
    Reads the dictionary of the detection and classification CNN
    and puts the parameters in the corresponding fields of the params object.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    dict_cnn : dict
        The keys are the parameter names and they are associated to their values.
    """

    # ResNet params
    params.L2_weight_decay = literal_eval(dict_resnet['L2_weight_decay'])
    params.batch_norm_decay = literal_eval(dict_resnet['batch_norm_decay'])
    params.batch_norm_epsilon = literal_eval(dict_resnet['batch_norm_epsilon'])
    # Adam
    params.learn_rate_adam = literal_eval(dict_resnet['learn_rate_adam'])
    params.beta_1 = literal_eval(dict_resnet['beta_1'])
    params.beta_2 = literal_eval(dict_resnet['beta_2'])
    params.epsilon = literal_eval(dict_resnet['epsilon'])
    params.batchsize = literal_eval(dict_resnet['batchsize'])
    # early stopping
    params.min_delta = literal_eval(dict_resnet['min_delta'])
    params.patience = literal_eval(dict_resnet['patience'])

def resnet_params_1(params, dict_resnet):
    """
    Reads the dictionary of the detection and classification CNN
    and puts the parameters in the corresponding fields of the params object.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    dict_cnn : dict
        The keys are the parameter names and they are associated to their values.
    """

    # ResNet params
    params.L2_weight_decay_1 = literal_eval(dict_resnet['L2_weight_decay'])
    params.batch_norm_decay_1 = literal_eval(dict_resnet['batch_norm_decay'])
    params.batch_norm_epsilon_1 = literal_eval(dict_resnet['batch_norm_epsilon'])
    # Adam
    params.learn_rate_adam_1 = literal_eval(dict_resnet['learn_rate_adam'])
    params.beta_1_1 = literal_eval(dict_resnet['beta_1'])
    params.beta_2_1 = literal_eval(dict_resnet['beta_2'])
    params.epsilon_1 = literal_eval(dict_resnet['epsilon'])
    params.batchsize_1 = literal_eval(dict_resnet['batchsize'])
    # early stopping
    params.min_delta_1 = literal_eval(dict_resnet['min_delta'])
    params.patience_1 = literal_eval(dict_resnet['patience'])

def resnet_params_2(params, dict_resnet):
    """
    Reads the dictionary of the detection and classification CNN
    and puts the parameters in the corresponding fields of the params object.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    dict_cnn : dict
        The keys are the parameter names and they are associated to their values.
    """

    # ResNet params
    params.L2_weight_decay_2 = literal_eval(dict_resnet['L2_weight_decay'])
    params.batch_norm_decay_2 = literal_eval(dict_resnet['batch_norm_decay'])
    params.batch_norm_epsilon_2 = literal_eval(dict_resnet['batch_norm_epsilon'])
    # Adam
    params.learn_rate_adam_2 = literal_eval(dict_resnet['learn_rate_adam'])
    params.beta_1_2 = literal_eval(dict_resnet['beta_1'])
    params.beta_2_2 = literal_eval(dict_resnet['beta_2'])
    params.epsilon_2 = literal_eval(dict_resnet['epsilon'])
    params.batchsize_2 = literal_eval(dict_resnet['batchsize'])
    # early stopping
    params.min_delta_2 = literal_eval(dict_resnet['min_delta'])
    params.patience_2 = literal_eval(dict_resnet['patience'])


def params_to_dict(params):
    """
    Converts the params object into a dictionary depending on the model.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    
    Returns
    --------
    dic : dict
        The keys are the parameter names and they are associated to their values.
    """
    
    dic = {}
    if params.classification_model not in ["cnn2", "resnet8", "resnet2", "hybrid_resnet_xgboost"]:
        dic["nb_conv_layers"] = params.nb_conv_layers
        dic["nb_dense_layers"] = params.nb_dense_layers
        dic["nb_filters"] = params.nb_filters 
        dic["filter_size"] = params.filter_size
        dic["pool_size"] = params.pool_size
        dic["nb_dense_nodes"] = params.nb_dense_nodes
        dic["dropout_proba"] = params.dropout_proba
    if params.classification_model not in ["cnn2", "resnet2"]:
        #Adam
        dic["learn_rate_adam"] = params.learn_rate_adam
        dic["beta_1"] = params.beta_1
        dic["beta_2"] = params.beta_2
        dic["epsilon"] = params.epsilon
        # early stopping
        dic["min_delta"] = params.min_delta
        dic["patience"] = params.patience
        # fit
        dic['batchsize'] = params.batchsize
    if params.classification_model in ["hybrid_cnn_xgboost", "hybrid_resnet_xgboost"]:
        dic["eta"] = params.eta
        dic["min_child_weight"] = params.min_child_weight
        dic["max_depth"] = params.max_depth
        dic["n_estimators"] = params.n_estimators
        dic["gamma_xgb"] = params.gamma_xgb
        dic["subsample"] = params.subsample
        dic["scale_pos_weight"] = params.scale_pos_weight
    if params.classification_model == "cnn2":
        dic["nb_conv_layers_1"] = params.nb_conv_layers_1
        dic["nb_dense_layers_1"] = params.nb_dense_layers_1
        dic["nb_filters_1"] = params.nb_filters_1
        dic["filter_size_1"] = params.filter_size_1
        dic["pool_size_1"] = params.pool_size_1
        dic["nb_dense_nodes_1"] = params.nb_dense_nodes_1
        dic["dropout_proba_1"] = params.dropout_proba_1
        dic["nb_conv_layers_2"] = params.nb_conv_layers_2
        dic["nb_dense_layers_2"] = params.nb_dense_layers_2
        dic["nb_filters_2"] = params.nb_filters_2 
        dic["filter_size_2"] = params.filter_size_2
        dic["pool_size_2"] = params.pool_size_2
        dic["nb_dense_nodes_2"] = params.nb_dense_nodes_2
        dic["dropout_proba_2"] = params.dropout_proba_2
    if params.classification_model in ["cnn2", "resnet2"]:
        dic["learn_rate_adam_1"] = params.learn_rate_adam_1
        dic["beta_1_1"] = params.beta_1_1
        dic["beta_2_1"] = params.beta_2_1
        dic["epsilon_1"] = params.epsilon_1
        dic["min_delta_1"] = params.min_delta_1
        dic["patience_1"] = params.patience_1
        dic['batchsize_1'] = params.batchsize_1
        dic["learn_rate_adam_2"] = params.learn_rate_adam_2
        dic["beta_1_2"] = params.beta_1_2
        dic["beta_2_2"] = params.beta_2_2
        dic["epsilon_2"] = params.epsilon_2
        dic["min_delta_2"] = params.min_delta_2
        dic["patience_2"] = params.patience_2
        dic['batchsize_2'] = params.batchsize_2
    if params.classification_model in ["resnet8", "hybrid_resnet_xgboost"]:
        dic['L2_weight_decay'] = params.L2_weight_decay
        dic['batch_norm_decay'] = params.batch_norm_decay
        dic['batch_norm_epsilon'] = params.batch_norm_epsilon
    if params.classification_model == "resnet2":
        dic['L2_weight_decay_1'] = params.L2_weight_decay_1
        dic['batch_norm_decay_1'] = params.batch_norm_decay_1
        dic['batch_norm_epsilon_1'] = params.batch_norm_epsilon_1
        dic['L2_weight_decay_2'] = params.L2_weight_decay_2
        dic['batch_norm_decay_2'] = params.batch_norm_decay_2
        dic['batch_norm_epsilon_2'] = params.batch_norm_epsilon_2
    return dic