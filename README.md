# 2022년 4학년 1학기 캡스톤디자인 최종프로젝트
- [월간 데이콘 신용카드 사용자 연체 예측 AI 경진대회](https://dacon.io/competitions/official/235713/overview/description)
- 신용카드 사용자 데이터를 사용자의 신용카드 대금 연체 정도를 예측하여 신용도를 분류하는 모델을 만든다.
- 이후 SHAP을 활용하여 모델의 결과를 해석하고 신뢰성을 확보하고자 한다.
- 연구는 데이터 수집 및 EDA - 분류 예측 모델 생성 - 모델 성능 평가 - SHAP을 활용한 모델 해석 순으로 진행한다.
- 분류 예측 모델의 생성을 위해 Logistic, Naïve Bayes, KNN, Decision Tree, Support vector machine, XGBoost, LightGBM, Catboost 총 8 개의 알고리즘을 사용한다. 이중 가장 성능이 좋은 알고리즘을 선정하여 XAI 분석을 실시하며 분석 방법으로는 SHAP 을 사용한다. SHAP 란 하나의 feature 에 대한 중요도를 알기 위해서, 여러 feature 들의 조합을 구성하고, 해당 feature 의 유무에 따른 평균적인 변화를 통해 값을 계산하는 방법이다.

## 모델선정
![image](https://user-images.githubusercontent.com/93179525/220556521-ded01dbc-3365-479f-9388-495fbf4d3ea7.png)
- LightGBM, XGBoost, Catboost 세가지 모델이 각 평가지표에서 근소한 차이로 상위를 차지함을 볼 수 있다. 따라서 해당 세가지 모델의 성능 보완 이후 재평가를 실시한다.
- 파생변수 생성(‘ID’는 ‘gender’, ‘DAYS_BIRTH’, ‘income_total’, ‘income_type’, ‘edu_type’, ‘occyp_type’ 값들을 더해서 만든
  파생변수) 및 K-prototype Clustering을 적용한 후 최종 모델로는 Catboost를 선정하였다.
- 해당 대회의 1등 모델과의 점수에 비해 Log Loss가 25% 개선됨을 확인할 수 있다.
![image](https://user-images.githubusercontent.com/93179525/220556450-05e99010-f266-4502-b100-8a8a8cbc202c.png)

## SHAP 분석
  ## 1. Feature Importance
    - Figure 19 를 통해 알 수 있듯이 ID 와 cluster 생성 전에는 Class 0,1,2 모두 ‘begin_month’가 모델 예측에
    제일 영향을 많이 끼치는 것을 알 수 있다. 영향을 가장 많이 주는 상위의 3 개 feature 인 ‘begin_month’, 
    ‘before_EMPLOTED_m’, ’DAYS_BIRTH_m’을 제외하고는 각 클래스별로 영향을 주는 feature 의 순위가 다른 것을
    파악할 수 있다.

    - 우리는 앞서 중복 데이터에 관한 이슈를 해결하기 위해 index 를 제외하고 완전히 일치하는 중복 데이터는
    삭제를 해주었다. 그럼에도 남아있던 중복 데이터를 구분 짓기 위해 주민등록번호와 같이 고유 식별력을 가진
    데이터가 필요했고, 이를 위해 ID 라는 컬럼을 생성해주었다. 부록의 Figure 20 에서 확인할 수 있듯이 ‘ID’와
    ‘cluster’의 생성 후에 X 축에 Shapely Value 값이 크게 증가하였고, Class 0,1,2 에서 모두 파생변수 ‘ID’가 모델에
    미치는 절대적인 영향도가 가장 큰 것을 확인할 수 있다.

Figure19 - ID, cluster 생성 전 SHAP Feature Importance
![image](https://user-images.githubusercontent.com/93179525/220558158-c4e5e7e7-a304-47df-a641-5455b789ae69.png)

Figure20 - ID, cluster 생성 후 SHAP Feature Importance
![image](https://user-images.githubusercontent.com/93179525/220558016-bac7be9b-e2b0-4e6f-95ff-4b4c6f080861.png)

  ## 2. Summary Plot
    - Summary plot 은 전체 특성들이 Shapley value 분포에 어떤 영향을 미치는지 시각화 한 것이다. X 축은 SHAP 
    값의 수치, Y 축은 독립변수, 색은 독립변수의 상대적인 크기를 나타낸다. 겹치는 점이 Y 축 방향으로 나타남에
    따라 독립변수에 관한 SHAP 값의 분포를 확인할 수 있다. SHAP 값의 양수와 음수는 예측 값에 긍정적인 기여와
    부정적인 기여를 각각 의미한다. 그래프의 색상이 빨간색이면 특성 값이 큼을 의미하고, 파란색이면 특성 값이
    작음을 의미한다. 그리고 회색은 categorical value 값이므로, 특성 값이 작고 큼을 나타낼 수 없기 때문에
    회색으로 나타난다. 부록의 그래프(Figure 21 에서 26)에서 ‘ID’, ‘occyp_type’, ’family_type’, ‘income_type’, ’reality’, 
    ‘edu_type’, ’house_type’은 categorical value 이므로 회색으로 나타난다.

    - 그래프에서 각 feature 의 순서는 예측에 미치는 영향력에 따라서 내림차순 정렬된다. Class 0,1,2 의 summary 
    plot 을살펴보면 ‘begin_month’가 정답을 판별하는데 가장 큰 기여를 했음을 확인할 수 있다.

Figure21 - ID, cluster 생성 전 Class 0 Summary Plot
![image](https://user-images.githubusercontent.com/93179525/220559060-389cf23b-a28a-472f-9c30-ba742c9a06b5.png)

Figure22 - ID, cluster 생성 전 Class 1 Summary Plot
![image](https://user-images.githubusercontent.com/93179525/220559190-4ec5e828-c692-4312-ab1c-1bee2c34e8dd.png)

Figure23 - ID, cluster 생성 전 Class 2 Summary Plot
![image](https://user-images.githubusercontent.com/93179525/220559233-d784a08b-d8c4-486b-a278-bdf6c803b1f0.png)

Figure24 - ID, cluster 생성 후 Class 0 Summary Plot
![image](https://user-images.githubusercontent.com/93179525/220559286-ab329ab5-0fcc-4f61-b723-f3966206e740.png)

Figure25 - ID, cluster 생성 후 Class 1 Summary Plot
![image](https://user-images.githubusercontent.com/93179525/220559404-8ad044dc-7a19-45c1-8d14-6ac62924920a.png)

Figure26 - ID, cluster 생성 후 Class 2 Summary Plot
![image](https://user-images.githubusercontent.com/93179525/220559431-a18781f9-9ba9-480f-a9e0-dafa894c9153.png)

![image](https://user-images.githubusercontent.com/93179525/220558776-65688ed3-d5be-47d2-a883-4dd20df4b178.png)

    - 수치형 변수의 신용등급에 따른 카드 발급 기간 차이를 파악하는 EDA 를 통해서 Class 2 에는 대부분
    ‘begin_month’ 값이 높은 사람들의 데이터가 분포했음을 확인할 수 있었다. 그에 비해서 Class 2 보다 신용등급이
    높은 Class 0 과 Class 1 은 ‘begin_month’ 값이 비교적 낮음을 알 수 있었다.
    Summary plot 에서 Class 0 과 1 은 ‘begin_month’의 특성 값이 작을수록 예측에 긍정적인 영향을 미치며, Class 
    2 는 ‘begin_month’의 특성 값이 클수록 모델의 예측에 긍정적인 영향을 미친다. 일반적으로는 카드를 발급하고
    난 기간이 길수록 신용도가 높을 것이라고 생각이 되지만 예상과는 달리 이렇게 정반대의 결과가 나타나는
    이유는 위의 그래프를 통해 확인했듯이 실제 데이터에서 ‘begin_month’의 값이 높은 데이터가 Class 2 에 많았고,
    그래서 모델은 ‘begin_month’가 비교적 높은 값은 Class 2 로 분류를 했다는 것을 Summary plot 을 통해 확인할
    수 있다.

    - ‘ID’는 ‘gender’, ‘DAYS_BIRTH’, ‘income_total’, ‘income_type’, ‘edu_type’, ‘occyp_type’ 값들을 더해서 만든
    파생변수이므로 categorical 한 column 이다. ID, cluster 생성 후에는 ID 의 feature importance 가 제일 높다. ‘ID’가
    모델의 정확도 향상에 유의미한 효과가 있었음을 확인할 수 있다.

  ## 3. Waterfall Plots
  - 생략(보고서 참고)
  
 ## 결론 및 기대효과
 - 지금까지 신용카드 이용 고객의 신용도를 예측하는 모델을 만들고 그 모델이 어느 정도의 설명력을 가지는지에
대한 분석 프로젝트를 모두 진행해보았다. 우리 팀에서 구현한 모델의 성능은 대회에 참여한 상위권 그룹의
결과를 바탕으로 성능을 비교해 봤을 때 더 우수한 성능을 가지는 것으로 확인되었으며. 또한 XAI 분석을
활용하여 우리 팀이 구축한 모델의 설명력과 신뢰성을 확보할 수 있는 과정도 진행하였다. 그러나 데이터 자체가
대회용 데이터이다 보니 노이즈가 존재하며, 실제 은행 데이터의 용량과 큰 차이가 있어서 실제 은행 고객의
데이터 특성을 완벽히 반영하지 못한 점에서 한계가 있었다. 하지만 우리가 도출해낸 일련의 과정들을 실제
금융업에서 사용하게 된다면, 고객에게는 자신의 신용도가 어떤 요인을 통해 도출되었는지 이해할 수 있는
객관적 근거를 마련하고 신용도에 문제가 있다면 문제를 파악할 수 있게 함으로써 그에 따른 해결방안을 생각할
수 있도록 한다. 은행에서는 자사 고객의 신용도를 분류하고 그 분류의 이유를 분석한 신뢰할 수 있는 결과를
토대로 기업 마케팅, 컨설팅, 금융상품 개발 등 다양한 측면에서 신뢰할 만한 인사이트를 제공할 것으로
기대된다.
