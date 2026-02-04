"""
Helper module for Optuna-based hyperparameter optimization.

This module provides utility functions for parameter suggestion and merging,
separated from the task runner to keep the runner classes concise.
"""
import copy


class OptunaHelper:
    """
    Helper class for Optuna hyperparameter optimization.
    """

    @staticmethod
    def suggest_param(trial, name: str, spec: list):
        """
        Convert search_space spec to Optuna suggest_* call.

        Args:
            trial: Optuna trial object.
            name: Parameter name.
            spec: List where first element is the method type and remaining elements
                  are method-specific arguments.
                  Supported methods:
                  - ['log_uniform', low, high]: Log-uniform float
                  - ['uniform', low, high]: Uniform float
                  - ['int', low, high]: Integer
                  - ['categorical', choices]: Categorical

        Returns:
            Suggested parameter value.

        Raises:
            ValueError: If the method type is unknown.
        """
        method = spec[0]
        if method == 'log_uniform':
            return trial.suggest_float(name, spec[1], spec[2], log=True)
        elif method == 'uniform':
            return trial.suggest_float(name, spec[1], spec[2])
        elif method == 'int':
            return trial.suggest_int(name, spec[1], spec[2])
        elif method == 'categorical':
            return trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f'Unknown search space method: {method}')

    @staticmethod
    def merge_params(base_params: dict | None, suggested: dict, search_space: dict) -> dict:
        """
        Merge base params with suggested params based on search_space keys.

        Args:
            base_params: Base parameters dict (can be None).
            suggested: Dict of suggested parameter values.
            search_space: Dict defining which parameters to merge from suggested.

        Returns:
            Merged parameters dict.
        """
        merged = copy.deepcopy(base_params) if base_params else {}
        for key in search_space:
            merged[key] = suggested[key]
        return merged

    @staticmethod
    def suggest_all_params(trial, search_space: dict) -> dict:
        """
        Suggest all parameters defined in search_space.

        Args:
            trial: Optuna trial object.
            search_space: Dict mapping parameter names to their specs.

        Returns:
            Dict of suggested parameter values.
        """
        suggested = {}
        for name, spec in search_space.items():
            suggested[name] = OptunaHelper.suggest_param(trial, name, spec)
        return suggested

    @staticmethod
    def build_params_for_trial(
        trial,
        search_space: dict,
        model_params: dict | None,
        optimizer_params: dict | None,
        batch_sampler_params: dict | None,
    ) -> tuple[dict, dict, dict]:
        """
        Build merged parameters for a trial.

        Suggests all parameters from search_space and merges them into the
        appropriate parameter dicts. Parameters not found in any of the base
        param dicts are added to model_params by default.

        Args:
            trial: Optuna trial object.
            search_space: Dict defining the hyperparameter search space.
            model_params: Base model parameters.
            optimizer_params: Base optimizer parameters.
            batch_sampler_params: Base batch sampler parameters.

        Returns:
            Tuple of (model_params, optimizer_params, batch_sampler_params).
        """
        suggested = OptunaHelper.suggest_all_params(trial, search_space)

        merged_model_params = OptunaHelper.merge_params(
            model_params, suggested,
            {k: v for k, v in search_space.items() if k in (model_params or {})}
        )
        merged_optimizer_params = OptunaHelper.merge_params(
            optimizer_params, suggested,
            {k: v for k, v in search_space.items() if k in (optimizer_params or {})}
        )
        merged_batch_sampler_params = OptunaHelper.merge_params(
            batch_sampler_params, suggested,
            {k: v for k, v in search_space.items() if k in (batch_sampler_params or {})}
        )

        for key in suggested:
            if key not in (merged_model_params or {}):
                if key not in (merged_optimizer_params or {}):
                    if key not in (merged_batch_sampler_params or {}):
                        merged_model_params[key] = suggested[key]

        return merged_model_params, merged_optimizer_params, merged_batch_sampler_params
