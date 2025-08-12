"""
Random Forest model for flood prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import structlog

from .base import BaseFloodModel

logger = structlog.get_logger()


class RandomForestFloodModel(BaseFloodModel):
    """
    Random Forest model for flood prediction
    
    Uses scikit-learn's RandomForestClassifier with hyperparameter optimization
    for flood event prediction. Provides interpretable predictions with feature
    importance analysis.
    """
    
    def __init__(
        self, 
        name: str = "RandomForestFlood",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: str = 'balanced',
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Random Forest flood model
        
        Parameters
        ----------
        name : str, optional
            Model name
        n_estimators : int, optional
            Number of trees (default: 100)
        max_depth : int, optional
            Maximum tree depth
        min_samples_split : int, optional
            Minimum samples required to split node
        min_samples_leaf : int, optional
            Minimum samples required in leaf node
        class_weight : str, optional
            Class weighting strategy ('balanced' handles imbalanced data)
        random_state : int, optional
            Random seed for reproducibility
        **kwargs
            Additional RandomForest parameters
        """
        super().__init__(name)
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight,
            'random_state': random_state,
            **kwargs
        }
        
        self.model = RandomForestClassifier(**self.params)
        self.cv_results = None
        
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        optimize_hyperparameters: bool = False,
        cv_folds: int = 3,
        **kwargs
    ) -> 'RandomForestFloodModel':
        """
        Fit Random Forest model to training data
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Binary target variable (0=no flood, 1=flood)
        optimize_hyperparameters : bool, optional
            Whether to perform hyperparameter optimization
        cv_folds : int, optional
            Number of CV folds for hyperparameter optimization
            
        Returns
        -------
        RandomForestFloodModel
            Fitted model instance
        """
        logger.info(
            f"Training {self.name} model", 
            samples=len(X), 
            features=len(X.columns),
            flood_events=y.sum()
        )
        
        self._validate_input(X, stage="training")
        self.feature_names = list(X.columns)
        
        # Store training info
        self.training_history['training_samples'] = len(X)
        self.training_history['num_features'] = len(X.columns)
        self.training_history['flood_events'] = int(y.sum())
        self.training_history['class_balance'] = y.value_counts().to_dict()
        
        # Prepare features
        X_prepared = self._prepare_features(X)
        
        if optimize_hyperparameters:
            self.model = self._optimize_hyperparameters(X_prepared, y, cv_folds)
        else:
            # Fit with current parameters
            self.model.fit(X_prepared, y)
        
        self.is_fitted = True
        
        # Store training results
        self.training_history['oob_score'] = getattr(self.model, 'oob_score_', None)
        
        logger.info(
            f"{self.name} training complete",
            oob_score=self.training_history.get('oob_score')
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict flood probability
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Flood probabilities (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self._validate_input(X, stage="prediction")
        X_prepared = self._prepare_features(X)
        
        # Return probability of flood class (class 1)
        probabilities = self.model.predict_proba(X_prepared)
        return probabilities[:, 1]  # Probability of class 1 (flood)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Class probabilities [P(no_flood), P(flood)]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self._validate_input(X, stage="prediction")
        X_prepared = self._prepare_features(X)
        
        return self.model.predict_proba(X_prepared)
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores
        
        Returns
        -------
        pd.Series
            Feature importance scores sorted by importance
        """
        if not self.is_fitted:
            logger.warning("Model must be fitted before getting feature importance")
            return None
        
        importance_scores = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
            name='importance'
        ).sort_values(ascending=False)
        
        return importance_scores
    
    def _optimize_hyperparameters(
        self, 
        X: np.ndarray, 
        y: pd.Series, 
        cv_folds: int
    ) -> RandomForestClassifier:
        """
        Optimize hyperparameters using GridSearchCV
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : pd.Series
            Target variable
        cv_folds : int
            Number of CV folds
            
        Returns
        -------
        RandomForestClassifier
            Optimized model
        """
        logger.info("Starting hyperparameter optimization")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Use ROC AUC as scoring metric (good for imbalanced data)
        grid_search = GridSearchCV(
            RandomForestClassifier(
                class_weight=self.params['class_weight'],
                random_state=self.params['random_state']
            ),
            param_grid=param_grid,
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Store CV results
        self.cv_results = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }
        
        self.training_history['hyperparameter_optimization'] = self.cv_results
        
        logger.info(
            "Hyperparameter optimization complete",
            best_score=grid_search.best_score_,
            best_params=grid_search.best_params_
        )
        
        return grid_search.best_estimator_
    
    def get_tree_depths(self) -> Dict[str, float]:
        """
        Get statistics about tree depths in the forest
        
        Returns
        -------
        dict
            Tree depth statistics
        """
        if not self.is_fitted:
            return {}
        
        depths = [tree.tree_.max_depth for tree in self.model.estimators_]
        
        return {
            'mean_depth': np.mean(depths),
            'median_depth': np.median(depths),
            'min_depth': np.min(depths),
            'max_depth': np.max(depths),
            'std_depth': np.std(depths)
        }
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """
        Get model complexity metrics
        
        Returns
        -------
        dict
            Model complexity information
        """
        if not self.is_fitted:
            return {}
        
        total_nodes = sum(tree.tree_.node_count for tree in self.model.estimators_)
        total_leaves = sum(tree.tree_.n_leaves for tree in self.model.estimators_)
        
        return {
            'n_estimators': self.model.n_estimators,
            'total_nodes': total_nodes,
            'total_leaves': total_leaves,
            'avg_nodes_per_tree': total_nodes / self.model.n_estimators,
            'avg_leaves_per_tree': total_leaves / self.model.n_estimators,
            'tree_depths': self.get_tree_depths()
        }
    
    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction using feature contributions
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        sample_idx : int, optional
            Index of sample to explain
            
        Returns
        -------
        dict
            Explanation of prediction
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explaining predictions")
        
        if sample_idx >= len(X):
            raise ValueError(f"Sample index {sample_idx} out of range")
        
        # Get prediction
        prediction = self.predict(X.iloc[[sample_idx]])[0]
        
        # Get feature values
        feature_values = X.iloc[sample_idx].to_dict()
        
        # Get feature importance
        importance = self.get_feature_importance()
        
        # Simple explanation based on feature importance and values
        explanation = {
            'prediction': prediction,
            'sample_features': feature_values,
            'top_features': importance.head(10).to_dict(),
            'model_info': {
                'name': self.name,
                'type': 'Random Forest',
                'n_estimators': self.model.n_estimators
            }
        }
        
        return explanation