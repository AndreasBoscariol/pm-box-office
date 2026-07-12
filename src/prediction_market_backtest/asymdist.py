"""Native asymmetric log-residual probability distributions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import t
from scipy.optimize import minimize

from .policy007 import WeightedPool,weighted_quantile


@dataclass(frozen=True,slots=True)
class SplitStudentT:
    location:float
    sigma_lower:float
    sigma_upper:float
    degrees_of_freedom:float

    def __post_init__(self)->None:
        if self.sigma_lower<=0 or self.sigma_upper<=0:raise ValueError("scales must be positive")
        if not 2.5<=self.degrees_of_freedom<=50:raise ValueError("degrees of freedom outside [2.5, 50]")
        ratio=self.sigma_upper/self.sigma_lower
        if not .5<=ratio<=5:raise ValueError("scale ratio outside [0.5, 5]")

    @property
    def lower_mass(self)->float:return self.sigma_lower/(self.sigma_lower+self.sigma_upper)

    def pdf(self,residual:float|Sequence[float])->np.ndarray:
        r=np.asarray(residual,float);scale=np.where(r<self.location,self.sigma_lower,self.sigma_upper);z=(r-self.location)/scale
        return 2*t.pdf(z,df=self.degrees_of_freedom)/(self.sigma_lower+self.sigma_upper)

    def cdf(self,residual:float|Sequence[float])->np.ndarray:
        r=np.asarray(residual,float);left=r<self.location;out=np.empty_like(r,dtype=float);total=self.sigma_lower+self.sigma_upper
        out[left]=2*self.sigma_lower/total*t.cdf((r[left]-self.location)/self.sigma_lower,df=self.degrees_of_freedom)
        out[~left]=self.lower_mass+2*self.sigma_upper/total*(t.cdf((r[~left]-self.location)/self.sigma_upper,df=self.degrees_of_freedom)-.5)
        return out

    def ppf(self,probability:float|Sequence[float])->np.ndarray:
        p=np.asarray(probability,float);lower=p<self.lower_mass;out=np.empty_like(p,dtype=float);total=self.sigma_lower+self.sigma_upper
        left_p=p[lower]*total/(2*self.sigma_lower);out[lower]=self.location+self.sigma_lower*t.ppf(left_p,df=self.degrees_of_freedom)
        right_p=.5+(p[~lower]-self.lower_mass)*total/(2*self.sigma_upper);out[~lower]=self.location+self.sigma_upper*t.ppf(right_p,df=self.degrees_of_freedom)
        return out

    def sample(self,n:int,seed:int)->np.ndarray:return self.ppf(np.random.default_rng(seed).random(n))


@dataclass(frozen=True,slots=True)
class AsymmetricEmpirical:
    residuals:np.ndarray
    weights:np.ndarray
    member_ids:tuple[int,...]
    delta:float=0
    lower_scale:float=1
    upper_scale:float=1
    upper_tail_stretch:float=1

    def __post_init__(self)->None:
        if not .5<=self.lower_scale<=3 or not .5<=self.upper_scale<=3:raise ValueError("empirical scales outside [0.5, 3]")
        if not 1<=self.upper_tail_stretch<=3:raise ValueError("tail stretch outside [1, 3]")
        if len(self.residuals)!=len(self.weights) or len(self.residuals)!=len(self.member_ids):raise ValueError("member identity and weights must align")

    @classmethod
    def from_pool(cls,pool:WeightedPool,**parameters:float)->"AsymmetricEmpirical":return cls(pool.residuals.copy(),pool.normalized_weights.copy(),pool.member_ids,**parameters)

    @property
    def transformed(self)->np.ndarray:
        median=float(weighted_quantile(self.residuals,self.weights,.5));values=self.delta+np.where(self.residuals<=median,self.lower_scale*(self.residuals-median),self.upper_scale*(self.residuals-median))
        q90=float(weighted_quantile(values,self.weights,.9));return np.where(values>q90,q90+self.upper_tail_stretch*(values-q90),values)

    def cdf(self,value:float)->float:return float(self.weights[self.transformed<=value].sum()/self.weights.sum())
    def ppf(self,probabilities:float|Sequence[float])->np.ndarray:return weighted_quantile(self.transformed,self.weights,probabilities)
    def sample(self,n:int,seed:int)->np.ndarray:return np.random.default_rng(seed).choice(self.transformed,n,p=self.weights/self.weights.sum())
    @property
    def membership_checksum(self)->str:return hashlib.sha256(np.asarray(self.member_ids,dtype="<i8").tobytes()).hexdigest()


def gross_cdf(distribution:SplitStudentT|AsymmetricEmpirical,threshold:float,point_forecast:float)->float:
    if threshold<=0:return 0.0
    return float(distribution.cdf(np.log(threshold/point_forecast)))
def gross_quantile(distribution:SplitStudentT|AsymmetricEmpirical,probabilities:float|Sequence[float],point_forecast:float)->np.ndarray:return point_forecast*np.exp(distribution.ppf(probabilities))


@dataclass(frozen=True,slots=True)
class ConditionalSplitStudentT:
    candidate:str;location:float;log_sigma_lower:float;log_sigma_upper:float;degrees_of_freedom:float
    lower_point_slope:float=0;upper_point_slope:float=0
    location_group_offsets:tuple[float,float,float]=(0,0,0)
    lower_group_offsets:tuple[float,float,float]=(0,0,0)
    upper_group_offsets:tuple[float,float,float]=(0,0,0)

    def distribution(self,point_forecast:float,origin:int)->SplitStudentT:
        group=origin_group_index(origin);x=np.log(point_forecast/20_000_000)
        lower=np.exp(self.log_sigma_lower+self.lower_point_slope*x+self.lower_group_offsets[group]);upper=np.exp(self.log_sigma_upper+self.upper_point_slope*x+self.upper_group_offsets[group])
        ratio=upper/lower
        if ratio<.5:upper=.5*lower
        elif ratio>5:upper=5*lower
        return SplitStudentT(self.location+self.location_group_offsets[group],lower,upper,self.degrees_of_freedom)


def fit_conditional_split_t(residuals:Sequence[float],points:Sequence[float],origins:Sequence[int],movie_ids:Sequence[int],candidate:str,penalty:float=5)->ConditionalSplitStudentT:
    from .probcal import movie_balanced_weights
    residuals=np.asarray(residuals,float);points=np.asarray(points,float);origins=np.asarray(origins,int);weights=movie_balanced_weights(movie_ids)
    dimensions={"global_split_student_t":4,"point_scale_split_student_t":6,"origin_point_scale_split_student_t":12}
    if candidate not in dimensions:raise ValueError(f"unknown candidate {candidate}")
    initial=np.zeros(dimensions[candidate]);initial[:4]=[float(np.median(residuals)),np.log(.35),np.log(.5),np.log(8-2.5)]
    bounds=[(-1,1),(np.log(.05),np.log(2)),(np.log(.05),np.log(2)),(np.log(.01),np.log(47.5))]+[(-.5,.5)]*(dimensions[candidate]-4)
    def unpack(params:np.ndarray)->ConditionalSplitStudentT:
        nu=2.5+np.exp(params[3]);lower_slope=upper_slope=0.;loc=(0.,0.,0.);lo=(0.,0.,0.);hi=(0.,0.,0.)
        if len(params)>=6:lower_slope,upper_slope=params[4:6]
        if len(params)==12:loc=(0.,params[6],params[7]);lo=(0.,params[8],params[9]);hi=(0.,params[10],params[11])
        return ConditionalSplitStudentT(candidate,float(params[0]),float(params[1]),float(params[2]),float(nu),float(lower_slope),float(upper_slope),tuple(map(float,loc)),tuple(map(float,lo)),tuple(map(float,hi)))
    def objective(params:np.ndarray)->float:
        model=unpack(params);groups=np.asarray([origin_group_index(value) for value in origins]);x=np.log(points/20_000_000)
        locations=model.location+np.asarray(model.location_group_offsets)[groups]
        lower=np.exp(model.log_sigma_lower+model.lower_point_slope*x+np.asarray(model.lower_group_offsets)[groups])
        upper=np.exp(model.log_sigma_upper+model.upper_point_slope*x+np.asarray(model.upper_group_offsets)[groups])
        ratio=upper/lower;upper=np.where(ratio<.5,.5*lower,np.where(ratio>5,5*lower,upper));ratios=np.log(upper/lower)
        scale=np.where(residuals<locations,lower,upper);z=(residuals-locations)/scale
        density=2*t.pdf(z,df=model.degrees_of_freedom)/(lower+upper)
        likelihood=-float(np.sum(weights*np.log(np.clip(density,1e-15,None))))
        regularized=params[4:] if len(params)>4 else np.array([])
        return likelihood+penalty*float(np.sum(regularized**2))+penalty*.1*float(np.mean(np.square(ratios)))
    result=minimize(objective,initial,method="L-BFGS-B",bounds=bounds,options={"maxiter":200})
    if not result.success:raise RuntimeError(f"{candidate} fit failed: {result.message}")
    return unpack(result.x)


def origin_group_index(origin:int)->int:return 0 if origin<=-8 else (1 if origin<=-3 else 2)
