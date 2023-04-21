# 한밭대학교 컴퓨터공학과 Attention팀

**팀 구성**
- 20191769 김지수 
- 20191767 김민지
- 20191730 민지민

## <u>Teamate</u> Project Background
- ### 필요성
  - 딥페이크는 사회적으로 악영향을 끼치는 기술로 딥페이크 사용의 96%는 포르노그라피로 사용됨
  - 심지어 전세계 딥페이크 피해자 중 25%가 한국인
- ### 기존 해결책의 문제점
  - 기존 딥페이크 탐지 모델은 간단한 이미지 편집 기법을 적용한 경우 탐지가 쉽게 무력화되는 문제점 발생함
- ### 제시하는 해결책
  - 다양한 유형의 공격들이 random하게 적용된 데이터셋을 안티포렌식 데이터셋으로 사용
  - 공격이 적용되지 않은 원본 데이터셋과 안티포렌식 데이터셋을 함께 학습에 적용하는 적대적 학습을 통해 학습에 적용한 공격 뿐만 아니라 학습에 적용하지 않은 공격에도 강인한 모델 개발
  
## System Design
  - ### System Requirements
    - 모델 도면
    
    ![image](https://user-images.githubusercontent.com/54435771/205571508-15d7ec61-8606-40cc-a490-ae638ccd01ba.png)
    
## Case Study
  - ### Description
  - 다양한 유형의 공격들을 random하게 적용한 안티포렌식 데이터 셋을 생성
  - 원본 데이터셋과 안티포렌식 데이터셋을 학습 데이터셋으로 사용하는 적대적 학습으로 최종 모델 도출
  
## Conclusion
  - ### 학습에 반영한 공격들 뿐만 아니라 학슴에 반영하지 않은 공격들에 대해서도 높은 탐지 강인성 확인
  - 정확도 측정 결과
  
  ![image](https://user-images.githubusercontent.com/54435771/205572001-1f8193ef-e8b6-4ea2-ba50-2d02d81a543d.png)

  - GUI
  
  ![image](https://user-images.githubusercontent.com/54435771/205572046-fdaeb26b-28d5-4ae3-95e3-7129a945d91c.png)

  
## Project Outcome
- 2022 정보처리학회(ASK) 춘계 학술대회 ‘학부생 논문경진대회’ 동상 수상
- 2022 한국디지털포렌식학회 하계 학술대회 경찰청장상 수상
