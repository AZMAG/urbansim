"""
Re-implementation of dcm.py

Supports MNL style choices.

"""
from __future__ import print_function, division

from collections import OrderedDict
import inspect

import numpy as np
import pandas as pd
from patsy import dmatrix
from prettytable import PrettyTable

from . import util
from ..utils import yamlio, choice
from ..urbanchoice import mnl


def get_mnl_utilities(data, model_expression, coeff):
    """
    Calculates MNL utilities for the provided interaction dataset.

    Parameters:
    -----------
    data: pandas.DataFrame
        Table containing interaction data.
    model_expression: str
        Patsy string defining the model specification.
    coeff: pandas.Series
        Coefficients to apply.

    Returns:
    --------
    pandas.Series with exponentiated utilities.

    """
    model_design = dmatrix(model_expression, data=data, return_type='dataframe')
    coeff = coeff.reindex(model_design.columns)

    coeff = np.reshape(np.array(coeff.values), (1, len(coeff)))
    model_data = np.transpose(model_design.values)
    utils = np.dot(coeff, model_data)
    exp_utils = np.exp(utils)

    return pd.Series(exp_utils.ravel(), data.index)


class MnlChoiceModel(object):
    """
    A discrete choice model with the ability to store an estimated
    model and predict new data based on the model.
    Based on multinomial logit.

    Parameters:
    -----------
    name : str
        Descriptive name for this model, used in output configuration files.
    outer: MnlChoiceModel, optional, default None
        Defines the `main` model that governs the overall model structure.
        If provided, this instance will serve as a sub-model.
        If not provided, this instance will serve as a main-model.
    choice_mode: str, default `individual`
        Defines the method used for making predictions, valid values
        are "individual" and "aggregate".
        `Individual` predictions will sample alterantives and compute utilities
        for each chooser, and each chooser will make a choice.
        `Aggregate` predictions will assign utilities direclty to alternatives,
        and then perform a single choice call, choosing `n` alternatives where `n`
        is the number of choosers. The chosen alternatives are then
        randomly assigned to the choosers. Therefore, expressions with chooser
        variables are not supported.
    constrained: bool, default True, not applicable for sub-models
        If True, predictions will be constained by the alternatives`
        capacity.
    model_expression: str, optional
        A patsy model expression. Should contain only a right-hand side.
    alts_choice_column: str, optional
        Name of the column in the `alternatives` table that choosers
        should choose. e.g. the 'building_id' column. If not provided
        the alternatives index is used.
    choosers_fit_size: numeric, optional
        Defines the number of choosers to use when fitting a model. If
        None, all choosers will be used.
        If less than 1, will be treated a percentage of choosers to sample.
    alts_fit_size: int, optional
        Defines the number of alternatives to sample per chooser
        when fitting a model. If None, all alterantives will be used.
    choosers_fit_filters, list of str, optional
        Filters applied to choosers table before fitting the model.
    alts_fit_filters : list of str, optional
        Filters applied to the alternatives table before fitting the model.
    alts_fit_sampling_weights_column: str, optional
        Column to use on alternatives table to influence sampling when
        fitting the model.
    alts_predict_size: int, optional
        Number of alternatives to sample when predicting.
    choosers_predict_filters: str, optional
        Filters applied to choosers before predicting.
    alts_predict_filters: str, optional
        Filters applied to alternatives before predicting.
    alts_predict_sampling_weights_column: str, optional
        Column to use on alternatives table to influence sampling when
        fitting the model.
    alts_predict_capacity_column: str, optional
        Column on alternatives table that defines the capacity (number
        of units available) for choosers. If None, each alternative
        assumed to have a single unit of capacity.
    predict_sampling_segmentation_column: str, optional
        Column used to define sampling segmentation for prediction. The
        column should reside on both choosers and alternatives.
    predict_sampling_within_percent: numeric, optional default 1
        If doing segmented sampling, defines the share of sampled alterantives
        that will come from the same segment as the chooser.
    predict_sampling_within_segments: OrderedDict, optional
        Allows for specifying sampling-within percentages for specific segments.
        Segments that are ommitted will revert to the default sampling-within
        percent argument.
    predict_normalize_unit_probs: bool, optional, default True
        Only applicable when performing an aggregate constrained prediction.
        If True, individual unit probabilities will be normalized so
        they sum to an alternative-level probability. For example, if a given
        alternative has a probabiity of .5, with 2 available units, the unit-level
        probabilities would be .25.
        If False, the probabilities will calaculated directly at the unit-level.
    predict_max_iterations:int, default 100
        Defines the maximum number of iterations when making constrained choices.
    sequential: bool, default False
        If the model has sub-models, and is constrained, defines how sub-models are evaluated.
        If True, each a sub-model will make choices, and remove capacities, before the
        next sub-model is evaluated.
        If False, sub-models are evaluated together, allowing choosers across sub-models
        to compete for the same alternatives.
    choosers_segmentation_column: str, optional
        Column on choosers table that will be used to assign choosers to
        a sub-model.
    fit_parameters: dict-like or pandas.DataFrame, optional, default None
        Results of a fit model. Should columns/keys for
        `Coefficient`, `Std. Error`, and `T-Score`.
    log_liklihoods: dict-like, optional, default None
        Log liklihood diagnostics for a fit model.
    """

    def __init__(self,
                 name,
                 outer=None,
                 choice_mode=None,
                 constrained=True,
                 model_expression=None,
                 alts_choice_column=None,
                 choosers_fit_size=None,
                 alts_fit_size=None,
                 choosers_fit_filters=None,
                 alts_fit_filters=None,
                 alts_fit_sampling_weights_column=None,
                 alts_predict_size=None,
                 choosers_predict_filters=None,
                 alts_predict_filters=None,
                 alts_predict_sampling_weights_column=None,
                 alts_predict_capacity_column=None,
                 predict_sampling_segmentation_column=None,
                 predict_sampling_within_percent=1,
                 predict_sampling_within_segments=None,
                 predict_normalize_unit_probs=True,
                 predict_max_iterations=100,
                 sequential=False,
                 choosers_segmentation_column=None,
                 fit_parameters=None,
                 log_likelihoods=None):

        # if this is the main model, check the choice mode
        if outer is None:
            valid_choice_modes = ['individual', 'aggregate']
            if choice_mode not in valid_choice_modes:
                raise ValueError('choice_mode must be one of {}'.format(valid_choice_modes))

        # maintain most of the model config as an internal dict
        local_vals = locals()
        local_ignore = ['self', 'outer']
        self._args = [a for a in inspect.getargspec(self.__init__).args if a not in local_ignore]
        self._args = OrderedDict((a, local_vals[a]) for a in self._args)

        # if fit parameters is a dictionary, convert to data frame
        if fit_parameters is not None:
            if not isinstance(fit_parameters, pd.DataFrame):
                self._config['fit_parameters'] = pd.DataFrame(fit_parameters)

        # list of model config properties not avilable to overriden by a sub-model
        self._main_args = [
            'choice_mode',
            'constrained',
            'alts_choice_column',
            'choosers_segmentation_column',
            'choosers_fit_filters',
            'choosers_predict_filters',
            'alts_predict_capacity_column',
            'predict_normalize_unit_probs',
            'predict_max_iterations',
            'sequential'
        ]

        # arguments that should be ignored based on another property
        self._ignore = {
            'individual': ['predict_normalize_unit_probs'],
            'aggregate': [
                'alts_predict_sampling_weights_column',
                'predict_sampling_within_percent',
                'predict_sampling_within_segments',
                'alts_predict_size',
            ],
            'sampling_segments': [
                'predict_sampling_within_percent',
                'predict_sampling_within_segments'
            ]
        }

        # set the outer/main model
        self._outer = outer

        # init sub models
        self.sub_models = OrderedDict()

    def add_sub_model(self,
                      name,
                      model_expression=None,
                      choosers_fit_size=None,
                      alts_fit_size=None,
                      alts_fit_filters=None,
                      alts_fit_sampling_weights_column=None,
                      alts_predict_size=None,
                      alts_predict_filters=None,
                      alts_predict_sampling_weights_column=None,
                      predict_sampling_segmentation_column=None,
                      predict_sampling_within_percent=None,
                      predict_sampling_within_segments=None,
                      fit_parameters=None,
                      log_likelihoods=None,):
        """
        Adds a sub-model configuration to the model.

        Parameters:
        -----------
        name : str
            Key for the model. Also used in output configuration files.
        model_expression: str, optional
            A patsy model expression. Should contain only a right-hand side.
        choosers_fit_size: numeric, optional
            Defines the number of choosers to use when fitting a model. If
            None, all choosers will be used.
            If less than 1, will be treated a percentage of choosers to sample.
        alts_fit_size: int, optional
            Defines the number of alternatives to sample per chooser
            when fitting a model. If None, all alterantives will be used.
        alts_fit_filters : list of str, optional
            Filters applied to the alternatives table before fitting the model.
        alts_fit_sampling_weights_column: str, optional
            Column to use on alternatives table to influence sampling when
            fitting the model.
        alts_predict_size: int, optional
            Number of alternatives to sample when predicting.
        alts_predict_filters: str, optional
            Filters applied to alternatives before predicting.
        alts_predict_sampling_weights_column: str, optional
            Column to use on alternatives table to influence sampling when
            fitting the model.
        predict_sampling_segmentation_column: str, optional
            Column used to define sampling segmentation for prediction. The
            column should reside on both choosers and alternatives.
        predict_sampling_within_percent: numeric, optional default 1
            If doing segmented sampling, defines the share of sampled alterantives
            that will come from the same segment as the chooser.
        predict_sampling_within_segments: OrderedDict, optional
            Allows for specifying sampling-within percentages for specific segments.
            Segments that are ommitted will revert to the default sampling-within
            percent argument.
        fit_parameters: dict-like or pandas.DataFrame, optional, default None
            Results of a fit model. Should columns/keys for
            `Coefficient`, `Std. Error`, and `T-Score`.
        log_liklihoods: dict-like, optional, default None
            Log liklihood diagnostics for a fit model.

        """
        # create an ordered dictionary from the provided args
        local_vals = locals()
        arg_names = inspect.getargspec(self.add_sub_model).args
        sub_model_args = OrderedDict((a, local_vals[a]) for a in arg_names if a != 'self')
        sub_model_args['outer'] = self

        # add the sub-model config to the main model
        self.sub_models[name] = MnlChoiceModel.from_dict(**sub_model_args)

    def _get_model_arg(self, key, inherit=True):
        """
        Used to dynamically retrieve a model argument.

        Parameters:
        -----------
        key: str
            Name of the argument to fetch.
        inherit: bool, optional, default True
            If True, a sub-model will defer to the main
            model property if the property is null or a main-model
            specific property. If False, returns None.

        """
        # handle choice mode-specicic properties
        if key != 'choice_mode':
            if key in self._ignore[self.choice_mode]:
                return None

        # handle sampling segmentation dependencies
        if key != 'predict_sampling_segmentation_column':
            if key in self._ignore['sampling_segments']:
                if self.predict_sampling_segmentation_column is None:
                    return None

        # get the attribute from the dictionary
        if key not in self._args:
            raise AttributeError('{} not found in model arguments'.format(key))
        a = self._args[key]

        # if sub-model inherit as needed
        if self.is_submodel:
            if key in self._main_args or a is None:
                if inherit:
                    return self._outer.__getattr__(key)
                else:
                    return None

        return a

    def __getattr__(self, key):
        """
        Allows access to model argument/configurations.

        """
        return self._get_model_arg(key)

    def __getitem__(self, key=None):
        """
        Returns an instance of sub-model, if the key is None,
        returns itself.

        """
        if key is None:
            return self
        else:
            return self.sub_models[key]

    def __setattr__(self, key, value):
        """
        Sets a value. If the key is in the model args/configuration dict this
        will be set there.

        """
        if hasattr(self, '_args'):
            if key in self._args:
                self._args[key] = value
                return

        self.__dict__[key] = value

    @property
    def is_submodel(self):
        """
        Indicates if this is a sub-model.

        """
        return self._outer is not None

    @property
    def has_submodels(self):
        """
        Indicates if this model contains sub-model definitions.

        """
        return len(self.sub_models) > 0

    @property
    def str_model_expression(self):
        """
        Model expression as a string suitable for use with patsy/statsmodels.

        """
        if self.model_expression is None:
            return None

        return util.str_model_expression(
            self.model_expression, add_constant=False)

    @classmethod
    def from_yaml(cls, yaml_str=None, str_or_buffer=None):
        """
        Create a MnlChoiceModel instance from a saved YAML configuration.
        Arguments are mutally exclusive.

        Parameters
        ----------
        yaml_str : str, optional
            A YAML string from which to load model.
        str_or_buffer : str or file like, optional
            File name or buffer from which to load YAML.

        Returns
        -------
        MnlChoiceModel

        """
        cfg = yamlio.yaml_to_dict(yaml_str, str_or_buffer, ordered=True)
        return MnlChoiceModel.from_dict(**cfg)

    @classmethod
    def from_dict(cls, **cfg):
        """
        Create a MnlChoiceModel instance from a saved dictionary.

        Parameters
        ----------
        **cfg: dict-like or keyword values

        Returns
        -------
        MnlChoiceModel

        """
        # create an instance of the model from the dictionary
        init_args = {a: cfg[a] for a in cfg if a != 'sub_models'}
        model = cls(**init_args)

        # add sub-models
        if 'sub_models' in cfg:
            for segment, sub_model_cfg in cfg['sub_models'].items():
                model.add_sub_model(segment, **sub_model_cfg)

        return model

    def to_dict(self, sparse=True):
        """
        Return a OrderedDict respresentation of an MnlChoiceModel
        instance.

        Parameters:
        -----------
        sparse: bool, optional, default True
            If True, parameters with null values will be ignored.

        """
        ignore = ['fit_parameters', 'log_likelihoods']
        if self.is_submodel:
            ignore.append('sequential')

        # get the model arguments, excluding the columns to ignore
        inherit = not sparse
        d = OrderedDict((a, self._get_model_arg(a, inherit)) for a in self._args if a not in ignore)
        if sparse:
            d = OrderedDict((k, v) for k, v in d.items() if v is not None)

        # get estimatation results
        if self.fit_parameters is not None:
            d['fit_parameters'] = yamlio.frame_to_yaml_safe(self.fit_parameters, True)
            d['log_likelihoods'] = self.log_likelihoods

        # get submodels
        if self.has_submodels:
            d['sub_models'] = {k: v.to_dict() for k, v in self.sub_models.items()}

        return d

    def to_yaml(self, str_or_buffer=None):
        """
        Save a model respresentation to YAML.

        Parameters
        ----------
        str_or_buffer : str or file like, optional
            By default a YAML string is returned. If a string is
            given here the YAML will be written to that file.
            If an object with a ``.write`` method is given the
            YAML will be written to that object.

        Returns
        -------
        j : str
            YAML is string if `str_or_buffer` is not given.

        """
        return yamlio.convert_to_yaml(self.to_dict(), str_or_buffer)

    def interaction_columns_used(self, segment=None):
        """
        Returns all columns used by the model expression.

        Parameters:
        ----------
        segment: value, optional, default None
            Optionally limit the columns to those used by a specific sub-model.

        Returns:
        --------
        List of column names

        """
        if segment is not None:
            return self.sub_models[segment].interaction_columns_used()

        # columns for the current model
        self_cols = util.columns_in_formula(self.model_expression)

        # strip _alt or _chooser from column names, these are
        # for cases where the same column exists on both choosers and alts
        self_cols = [x.replace('_chooser', '') for x in self_cols]
        self_cols = [x.replace('_alt', '') for x in self_cols]
        self_cols = set(self_cols)

        # columns for all sub-models
        sm_cols = []
        if self.has_submodels:
            sm_cols = [sm.interaction_columns_used() for sm in self.sub_models.values()]

        return list(self_cols.union(*sm_cols))

    def choosers_columns_used(self, segment=None, choosers=None, fit=True, predict=True):
        """
        Columns from the choosers table used in the model.

        Parameters:
        ----------
        segment: value, optional, default None
            Optionally limit the columns to those used by a specific sub-model.
        choosers: pandas.DataFrame or table-like optional, default None
            If provided, the result will include columns from used by
            the model expression (i.e. all model columns).
            If not provided, this is limited to columns used by filters.
        fit: bool, optional, default True
            If False, columns only needed for fitting are ignored.
        predict: bool, optional, default True
            If False, columns only needed for predicting are ignored.

        Returns:
        --------
        List of column names

        """
        if segment is not None:
            return self.sub_models[segment].choosers_columns_used(None, choosers, fit, predict)

        # get filter cols in the current model
        self_cols = set([self.choosers_segmentation_column])

        if fit:
            self_cols = self_cols.union(
                set(util.columns_in_filters(self.choosers_fit_filters)))

        if predict:
            self_cols.add(self.predict_sampling_segmentation_column)
            self_cols = self_cols.union(
                set(util.columns_in_filters(self.choosers_predict_filters)))

        # get expression columns for the current model
        if choosers is not None:
            exp_cols_all = self.interaction_columns_used()
            exp_cols = [c for c in exp_cols_all if c in choosers.columns]
            self_cols = self_cols.union(set(exp_cols))

        # get sub-model columns
        sm_cols = set()
        if self.has_submodels:
            sm_cols = [sm.choosers_columns_used(
                None, choosers, fit, predict)for sm in self.sub_models.values()]

        all_cols = self_cols.union(*sm_cols)
        return [c for c in all_cols if c is not None]

    def alts_columns_used(self, segment=None, alternatives=None, fit=True, predict=True):
        """
        Columns from the alternatives table used in the model.

        Parameters:
        ----------
        segment: value, optional, default None
            Optionally limit the columns to those used by a specific sub-model.
        alternatives: pandas.DataFrame or table-like optional, default None
            If provided, the result will include columns from used by
            the model expression (i.e. all model columns).
            If not provided, this is limited to columns used by filters.
        fit: bool, optional, default True
            If False, columns only needed for fitting are ignored.
        predict: bool, optional, default True
            If False, columns only needed for predicting are ignored.

        Returns:
        --------
        List of column names

        """
        if segment is not None:
            return self.sub_models[segment].alts_columns_used(None, alternatives, fit, predict)

        # get filters in the current model
        self_cols = set()
        if fit:
            self_cols.add(self.alts_fit_sampling_weights_column)
            self_cols = self_cols.union(
                set(util.columns_in_filters(self.alts_fit_filters)))

        if predict:
            self_cols.add(self.predict_sampling_segmentation_column)
            self_cols.add(self.alts_predict_sampling_weights_column)
            self_cols.add(self.alts_predict_capacity_column)
            self_cols = self_cols.union(
                set(util.columns_in_filters(self.alts_predict_filters)))

        # get expression columns for the current model
        if alternatives is not None:
            exp_cols_all = self.interaction_columns_used()
            exp_cols = [c for c in exp_cols_all if c in alternatives.columns]

            self_cols = self_cols.union(set(exp_cols))

        # get filters in sub-models
        sm_cols = []
        if self.has_submodels:
            sm_cols = [sm.alts_columns_used(
                None, alternatives, fit, predict) for sm in self.sub_models.values()]

        all_cols = self_cols.union(*sm_cols)
        return [c for c in all_cols if c is not None]

    def columns_used(self, segment=None, fit=True, predict=True):
        """
        Columns from any table used in the model. May come from either
        the choosers or alternatives tables.

        Parameters:
        ----------
        segment: value, optional, default None
            The segment of the sub-model to get columns for.
        fit: bool, optional, default True
            If False, columns only needed for fitting are ignored.
        predict: bool, optional, default True
            If False, columns only needed for predicting are ignored.

        """
        return list(set().union(*[
            self.choosers_columns_used(segment, fit=fit, predict=predict),
            self.alts_columns_used(segment, fit=fit, predict=predict),
            self.interaction_columns_used(segment)
        ]))

    def report_fit(self):
        """
        Print a report of the fit results.

        """
        if self.fit_parameters is None:
            print('Model not yet fit.')
            return

        print('Null Log-liklihood: {0:.3f}'.format(
            self.log_likelihoods['null']))
        print('Log-liklihood at convergence: {0:.3f}'.format(
            self.log_likelihoods['convergence']))
        print('Log-liklihood Ratio: {0:.3f}\n'.format(
            self.log_likelihoods['ratio']))

        tbl = PrettyTable(
            ['Component', ])
        tbl = PrettyTable()

        tbl.add_column('Component', self.fit_parameters.index.values)
        for col in ('Coefficient', 'Std. Error', 'T-Score'):
            tbl.add_column(col, self.fit_parameters[col].values)

        tbl.align['Component'] = 'l'
        tbl.float_format = '.3'

        print(tbl)

    def fit(self, choosers, alternatives, current_choice):
        """
        Fit and save model parameters based on given data. Will update the
        `fit_parameters` and `log_likelihoods` properties on the model or
        sub-models.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing,
            e.g. buildings.
        current_choice : pandas.Series or any
            A Series describing the `alternatives` currently chosen
            by the `choosers`. Should have an index matching `choosers`
            and values matching the index of `alternatives`.
            If a non-Series is given it should be a column in `choosers`.

        """

        # make sure alternative IDs are on the index
        if self.alts_choice_column is not None:
            alternatives = alternatives.set_index(self.alts_choice_column, drop=False)

        # apply upper/main level filters
        if self.is_submodel:
            c_filters = self._outer.choosers_fit_filters
            a_filters = self._outer.alts_fit_filters
        else:
            c_filters = self.choosers_fit_filters
            a_filters = self.alts_fit_filters
        choosers = util.apply_filter_query(choosers, c_filters)
        alternatives = util.apply_filter_query(alternatives, a_filters)

        # if chooser segmentation is defined, but there are no sub-models, add a sub-model
        # for each unique value
        if self.choosers_segmentation_column is not None and not self.has_submodels:
            for s in choosers[self.choosers_segmentation_column].unique():
                if isinstance(s, unicode):
                    s = str(s)
                else:
                    s = np.asscalar(s)

                print('adding {}...'.format(s))

                self.add_sub_model(s)

        # inner function to handle the fit for a given model
        def fit_model(key, model, choosers, alternatives, current_choice):
            print('fitting {}...'.format(key))

            # apply sub-model filters
            if model.is_submodel:
                if model.choosers_segmentation_column is not None:
                    f = "{} == '{}'" if isinstance(key, str) else '{} == {}'
                    f = f.format(model.choosers_segmentation_column, key)
                    choosers = util.apply_filter_query(choosers, f)

                alternatives = util.apply_filter_query(alternatives, model.alts_fit_filters)

            # get observed choicess
            if isinstance(current_choice, pd.Series):
                current_choice = current_choice.loc[choosers.index]
            else:
                current_choice = choosers[current_choice]

            # just keep alternatives who have chosen an available alternative
            in_alts = current_choice.isin(alternatives.index)
            choosers = choosers[in_alts]
            current_choice = current_choice[in_alts]

            # just keep interaction columns
            choosers = choosers[model.choosers_columns_used(None, choosers, False, False)]
            alternatives = alternatives[model.alts_columns_used(None, alternatives, False, False)]

            # sample choosers for estimation
            num_choosers = model.choosers_fit_size

            if num_choosers is not None:
                if num_choosers < 1:
                    # the parameter is expressed as a percentage of the available choosers
                    num_choosers = 1 + int(len(choosers) * num_choosers)
                num_choosers = min(num_choosers, len(choosers))
                idx = np.random.choice(choosers.index, num_choosers, replace=False)
                choosers = choosers.loc[idx]
                current_choice = current_choice.loc[idx]

            # get interaction data
            if model.choice_mode == 'aggregate':
                # don't allow chooser variables if doing aggregate predictions
                choosers = pd.DataFrame(index=choosers.index)
            sample_size = self.alts_fit_size
            print ('alts to sample: {}'.format(sample_size))
            print ('num choosers: {}'.format(num_choosers))

            sampling_weights = model.alts_fit_sampling_weights_column
            interact, sample_size, chosen = choice.get_interaction_data_for_estimation(
                choosers, alternatives, current_choice, sample_size, sampling_weights)

            # get the design matrix
            model_design = dmatrix(
                model.str_model_expression, data=interact, return_type='dataframe')

            # estimate and report
            # TODO: make the estimation method a callback provided as a model argument
            log_likelihoods, fit_parameters = mnl.mnl_estimate(
                model_design.values, chosen, sample_size)
            fit_parameters.index = model_design.columns

            model.fit_data = interact
            model.fit_chosen = chosen
            model.log_likelihoods = log_likelihoods
            model.fit_parameters = fit_parameters

            model.report_fit()
            # logger.debug('finish: fit DCM model {}'.format(name))

        # get the collection of models to fit
        models = {None: self}
        if self.has_submodels:
            models = self.sub_models

        for key, model in models.items():
            fit_model(key, model, choosers, alternatives, current_choice)

    @property
    def fitted(self):
        """
        True if model is ready for prediction.

        """
        if self.has_submodels:
            return all([m.fit_parameters is not None for m in self.sub_models.values()])
        else:
            return self.fit_parameters is not None

    def assert_fitted(self):
        """
        Raises `RuntimeError` if the model is not ready for prediction.

        """
        if not self.fitted:
            raise RuntimeError('Model has not been fit.')

    def predict(self, choosers, alternatives, debug=False,
                constrained=None, max_iterations=None, sequential=None):
        """
        Make choice predictions using fitted results.

        Parameters:
        -----------
        choosers: pandas.DataFrame
            Agents making choices.
        alternatives: pandas.DataFrame
            Choice set to choose from.
        debug: optional, default False
            If True, diagnostic information about the alternatives,
            utilities and probabilities will be returned.
        constrained: bool, optional, default None
            If provided, will temporarily override the `constrained` property.
        max_iterations: int, optional, default None
            If provided, will temporarily override the `predict_max_iterations` property.
        sequential: bool, optional, default None
            If provided, will temporarily override the `sequential` property.

        Returns:
        -------
        choices: pandas.Series

        capacities: pandas.Series

        verbose: pandas.DataFrame

        """

        # handle ad-hoc overrides of model inputs
        if constrained is None:
            constrained = self.constrained

        if max_iterations is None:
            max_iterations = self.predict_max_iterations

        if sequential is None:
            sequential = self.sequential

        # make sure alternative IDs are on the index
        if self.alts_choice_column is not None:
            alternatives = alternatives.set_index(self.alts_choice_column, drop=False)

        # apply upper/main level filters
        if self.is_submodel:
            c_filters = self._outer.choosers_predict_filters
            a_filters = self._outer.alts_predict_filters
        else:
            c_filters = self.choosers_predict_filters
            a_filters = self.alts_predict_filters
        choosers = util.apply_filter_query(choosers, c_filters)
        alternatives = util.apply_filter_query(alternatives, a_filters)

        # if sub-modeled, only keep choosers in a valid segment
        if self.has_submodels:
            segments = np.array(self.sub_models.keys())
            keep = choosers[self.choosers_segmentation_column].isin(segments)
            choosers = choosers[keep]

        # get the alternative capacities
        capacities = None
        if constrained:
            capacities = choice.get_capacities(
                alternatives, self.alts_predict_capacity_column)

        # get the collection of models to predict
        models = {None: self}
        if self.has_submodels:
            models = self.sub_models

        # if running in a constrained AND sequential manner, remove capacities
        # after each sub-model; if not running sequentially leave the removal of
        # capacities to the constrained choice method
        if constrained and sequential:
            remove_cap = True
        else:
            remove_cap = False

        # callback for making constrained choices across all models
        def main_callback(choosers, alternatives, verbose, capacities, as_callback=True):

            choices_all = []
            verbose_all = []

            for key, model in models.items():

                print(key)

                # apply sub-model filters
                if model.is_submodel:
                    if model.choosers_segmentation_column is not None:
                        f = "{} == '{}'" if isinstance(key, str) else '{} == {}'
                        f = f.format(model.choosers_segmentation_column, key)
                        curr_choosers = util.apply_filter_query(choosers, f)

                    curr_alts = util.apply_filter_query(alternatives, model.alts_predict_filters)

                # reindex the capacities to match the alternatives
                curr_cap = None
                if constrained:
                    curr_cap = capacities.reindex(alternatives.index)

                # define the weights callback (generates utilities)
                def weights_cb(interaction_data):
                    return get_mnl_utilities(
                        interaction_data,
                        model.str_model_expression,
                        model.fit_parameters['Coefficient']
                    )

                # make the choice
                if model.choice_mode == 'individual':
                    choice_cb = choice.IndividualChoiceCallback(
                        weights_cb,
                        model.predict_sampling_segmentation_column,
                        model.alts_predict_size,
                        model.alts_predict_sampling_weights_column,
                        model.predict_sampling_within_percent,
                        model.predict_sampling_within_segments
                    )
                    if remove_cap:
                        c, capacities, v = choice.constrained_choice(
                            curr_choosers,
                            curr_alts,
                            choice_cb,
                            capacity=curr_cap,
                            verbose=verbose,
                            max_iterations=max_iterations
                        )
                    else:
                        c, v = choice_cb(curr_choosers, curr_alts, verbose)

                else:
                    c, v = choice.aggregate_choice(
                        curr_choosers,
                        curr_alts,
                        weights_cb,
                        constrained,
                        model.predict_sampling_segmentation_column,
                        curr_cap,
                        model.predict_normalize_unit_probs
                    )

                    if remove_cap and c is not None:
                        capacities -= c['alternative_id'].value_counts().reindex(
                            capacities.index).fillna(0)

                if c is not None:
                    choices_all.append(c)

                if v is not None and verbose:
                    verbose_all.append(v)

            # combine results across models
            choices_concat = pd.concat(choices_all) if len(choices_all) > 0 else None
            verbose_concat = pd.concat(verbose_all) if len(verbose_all) > 0 else None

            if as_callback:
                return choices_concat, verbose_concat
            else:
                return choices_concat, capacities, verbose_concat

        # main execution
        if sequential or not constrained:
            # run each sub-model in order and allow it to fully complete
            choices, cap, pdf = main_callback(
                choosers, alternatives, debug, capacities, as_callback=False)
            choices = choices.reindex(choosers.index)
        else:
            # allow choosers across models to compete for the same alternatives
            choices, cap, pdf = choice.constrained_choice(
                choosers,
                alternatives,
                main_callback,
                capacity=capacities,
                max_iterations=max_iterations,
                verbose=debug
            )

        # now do something else?
        self.sim_pdf = pdf
        return choices, cap, pdf
