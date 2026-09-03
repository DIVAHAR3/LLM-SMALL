from typing import Optional, Union

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text to continue")
    max_new_tokens: int = Field(default=200, ge=1, le=500, description="Tokens to generate")
    temperature: float = Field(default=1.0, gt=0, le=5.0, description="Sampling temperature")
    top_k: Optional[int] = Field(default=None, ge=1, description="Keep only the top_k most likely tokens")
    top_p: Optional[float] = Field(default=None, gt=0, le=1.0, description="Nucleus sampling threshold")
    greedy: bool = Field(default=False, description="Always pick the single most likely token")


class GenerateResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    checkpoint: str
    params: int


class BrightnessStats(BaseModel):
    mean: float
    stddev: float


class DominantColor(BaseModel):
    hex: str
    percent: float


class ImageAnalysisResponse(BaseModel):
    format: str
    mode: str
    width: int
    height: int
    aspect_ratio: float
    size_bytes: int
    megapixels: float
    brightness: BrightnessStats
    dominant_colors: list[DominantColor]
    exif: dict[str, Union[str, int, float]]
