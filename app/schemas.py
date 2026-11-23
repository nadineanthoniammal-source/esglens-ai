"""Pydantic schema for structured ESG extraction."""
from typing import Optional
from pydantic import BaseModel, Field


class Metric(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None
    year: Optional[str] = None
    page: Optional[int] = None  # page number the value was found on
    derived: bool = False  # True if computed from other extracted values


class ESGMetrics(BaseModel):
    company_name: Optional[str] = None
    reporting_year: Optional[str] = None
    scope1_emissions: Metric = Field(default_factory=Metric)
    scope2_emissions: Metric = Field(default_factory=Metric)
    scope3_emissions: Metric = Field(default_factory=Metric)
    total_energy_consumption: Metric = Field(default_factory=Metric)
    renewable_energy_consumption: Metric = Field(default_factory=Metric)
    renewable_energy_share_percent: Metric = Field(default_factory=Metric)
    water_consumption: Metric = Field(default_factory=Metric)
    waste_generated: Metric = Field(default_factory=Metric)
    net_zero_target_year: Optional[str] = None


METRIC_KEYS = [
    "scope1_emissions", "scope2_emissions", "scope3_emissions",
    "total_energy_consumption", "renewable_energy_consumption", "renewable_energy_share_percent",
    "water_consumption", "waste_generated",
]
