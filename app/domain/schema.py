from pydantic import BaseModel, Field

FEATURES = [
    "amount",
    "hour",
    "distance_km",
    "transactions_24h",
    "account_age_days",
    "international",
]


class Transaction(BaseModel):
    amount: float = Field(ge=0, le=1000000)
    hour: int = Field(ge=0, le=23)
    distance_km: float = Field(ge=0, le=20000)
    transactions_24h: int = Field(ge=0, le=1000)
    account_age_days: int = Field(ge=0, le=20000)
    international: int = Field(ge=0, le=1)
