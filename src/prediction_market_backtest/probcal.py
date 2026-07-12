"""Market-independent probability recalibration for 007 residual distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import betainc, betaincinv

from .cdf_validation import CDFCandidate


class CalibrationMap(Protocol):
    def transform(self, probabilities: float | Sequence[float]) -> np.ndarray: ...
    def inverse(self, probabilities: float | Sequence[float]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class BetaCalibration:
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0: raise ValueError("beta parameters must be positive")

    def transform(self, probabilities: float | Sequence[float]) -> np.ndarray:
        return betainc(self.alpha,self.beta,np.clip(np.asarray(probabilities,float),0,1))

    def inverse(self, probabilities: float | Sequence[float]) -> np.ndarray:
        return betaincinv(self.alpha,self.beta,np.clip(np.asarray(probabilities,float),0,1))


@dataclass(frozen=True, slots=True)
class PowerCalibration:
    exponent: float

    def __post_init__(self) -> None:
        if not .5 <= self.exponent <= 2: raise ValueError("power exponent outside [0.5, 2]")

    def transform(self, probabilities: float | Sequence[float]) -> np.ndarray:
        return np.power(np.clip(np.asarray(probabilities,float),0,1),self.exponent)

    def inverse(self, probabilities: float | Sequence[float]) -> np.ndarray:
        return np.power(np.clip(np.asarray(probabilities,float),0,1),1/self.exponent)


@dataclass(frozen=True, slots=True)
class OriginGroupPowerCalibration:
    exponents: tuple[float,float,float]
    group_index: int

    def transform(self, probabilities: float | Sequence[float]) -> np.ndarray:
        return PowerCalibration(self.exponents[self.group_index]).transform(probabilities)
    def inverse(self, probabilities: float | Sequence[float]) -> np.ndarray:
        return PowerCalibration(self.exponents[self.group_index]).inverse(probabilities)


@dataclass(frozen=True, slots=True)
class ShrunkEmpiricalCalibration:
    pit_values: np.ndarray
    weights: np.ndarray
    shrinkage: float

    def __post_init__(self) -> None:
        if not 0<=self.shrinkage<=1:raise ValueError("shrinkage must be in [0,1]")
        if len(self.pit_values)!=len(self.weights) or not len(self.pit_values):raise ValueError("aligned PIT values and weights required")

    def transform(self, probabilities: float | Sequence[float]) -> np.ndarray:
        u=np.asarray(probabilities,float);order=np.argsort(self.pit_values);p=np.asarray(self.pit_values)[order];w=np.asarray(self.weights,float)[order];cdf=np.cumsum(w)/w.sum()
        empirical=np.interp(u,np.r_[0,p,1],np.r_[0,cdf,1]);return (1-self.shrinkage)*u+self.shrinkage*empirical

    def inverse(self, probabilities: float | Sequence[float]) -> np.ndarray:
        grid=np.linspace(0,1,10001);mapped=self.transform(grid);return np.interp(np.asarray(probabilities,float),mapped,grid)


@dataclass(frozen=True, slots=True)
class CalibratedDistribution:
    name: str
    base: CDFCandidate
    calibration: CalibrationMap | None = None
    location_shift_log: float = 0.0

    def cdf(self,outcome:float,point:float)->float:
        base_probability=self.base.cdf(outcome/(np.exp(self.location_shift_log)),point)
        return float(self.calibration.transform(base_probability)) if self.calibration else base_probability

    def quantile(self,probabilities:float|Sequence[float],point:float)->np.ndarray:
        p=np.asarray(probabilities,float);base_p=self.calibration.inverse(p) if self.calibration else p
        return self.base.quantile(base_p,point)*np.exp(self.location_shift_log)


def movie_balanced_weights(movie_ids:Sequence[int])->np.ndarray:
    ids=np.asarray(movie_ids);unique,counts=np.unique(ids,return_counts=True);lookup=dict(zip(unique,counts));weights=np.asarray([1/lookup[value] for value in ids],float);return weights/weights.sum()


def fit_beta_calibration(pit_values:Sequence[float],movie_ids:Sequence[int],regularization:float=5.0)->BetaCalibration:
    pits=np.clip(np.asarray(pit_values,float),1e-6,1-1e-6);weights=movie_balanced_weights(movie_ids)
    def objective(log_params:np.ndarray)->float:
        alpha,beta=np.exp(log_params);log_density=(alpha-1)*np.log(pits)+(beta-1)*np.log1p(-pits)
        from scipy.special import betaln
        likelihood=-float(np.sum(weights*(log_density-betaln(alpha,beta))))
        return likelihood+regularization*float(np.sum(log_params**2))
    result=minimize(objective,np.zeros(2),method="L-BFGS-B",bounds=[(-3,3),(-3,3)])
    if not result.success:raise RuntimeError("beta calibration fit failed")
    alpha,beta=np.exp(result.x);return BetaCalibration(float(alpha),float(beta))


def fit_power_calibration(pit_values:Sequence[float],movie_ids:Sequence[int],regularization:float=5.0)->PowerCalibration:
    pits=np.clip(np.asarray(pit_values,float),1e-8,1);weights=movie_balanced_weights(movie_ids)
    def objective(log_a:np.ndarray)->float:
        a=float(np.exp(log_a[0]));return -float(np.sum(weights*(np.log(a)+(a-1)*np.log(pits))))+regularization*float(log_a[0]**2)
    result=minimize(objective,np.zeros(1),method="L-BFGS-B",bounds=[(np.log(.5),np.log(2))])
    if not result.success:raise RuntimeError("power calibration fit failed")
    return PowerCalibration(float(np.exp(result.x[0])))


def fit_origin_group_power(pit_values:Sequence[float],movie_ids:Sequence[int],groups:Sequence[int],regularization:float=5.0)->tuple[float,float,float]:
    pits=np.clip(np.asarray(pit_values,float),1e-8,1);groups=np.asarray(groups,int);weights=movie_balanced_weights(movie_ids)
    def objective(params:np.ndarray)->float:
        global_log=params[0];logs=global_log+np.r_[0,params[1:3]];a=np.exp(logs[groups]);likelihood=-float(np.sum(weights*(np.log(a)+(a-1)*np.log(pits))));return likelihood+regularization*float(np.sum(params[1:]**2)+.25*global_log**2)
    result=minimize(objective,np.zeros(3),method="L-BFGS-B",bounds=[(np.log(.5),np.log(2))]*3)
    if not result.success:raise RuntimeError("origin power calibration fit failed")
    logs=result.x[0]+np.r_[0,result.x[1:3]];return tuple(float(value) for value in np.exp(np.clip(logs,np.log(.5),np.log(2))))


def fit_isotonic_calibration(pit_values:Sequence[float],movie_ids:Sequence[int],shrinkage:float)->ShrunkEmpiricalCalibration:
    return ShrunkEmpiricalCalibration(np.asarray(pit_values,float),movie_balanced_weights(movie_ids),shrinkage)


def location_shift(log_errors:Sequence[float],movie_ids:Sequence[int])->float:
    values=np.asarray(log_errors,float);weights=movie_balanced_weights(movie_ids);order=np.argsort(values);values,weights=values[order],weights[order]
    return float(values[np.searchsorted(np.cumsum(weights),.5,side="left")])


def assert_chronological(training_dates:Sequence[object],test_date:object)->None:
    if any(date>=test_date for date in training_dates):raise ValueError("calibration history contains current or future movie")
