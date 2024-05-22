# DCN-V2 PyTorch (In development)

[English](docs/README_EN.md)

논문 출처: [[Wang et al. 2020: DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems]](https://arxiv.org/pdf/2008.13535)

Web 환경에서 얻을 수 있는 다양한 형태의 특성간 관계를 Polynomial Approximation을 통해서 훈련하여 추천 시스템, CTR 예측 등에 활용하는 DCN-V2을 PyTorch Lightning을 이용해 구현하는 프로젝트

## TO-DO List

- DCN-V2 torch.nn.Module 클래스 구현
- LightningModule 클래스 구현
- DataModule 클래스 구현
    - pandas, polars support
    - scikit-learn pipeline support
- yaml 파일을 이용한 모델/훈련 하이퍼파라미터 커스터마이징 구현
- 훈련/검증 스크립트 구현
- MovieLens 벤치마크 성능 재현

