# ModelAI

캡스톤 디자인 코난 팀 입니다.

코난 목소리로 변조하는 모델입니다.

git commit -m "comment: a2c코드에 코멘트 추가#1"

git remote update

python train.py --dir llvc_nc # 파이썬 코드 실행 방법 (꼭 --dir 인자를 포함해서 실행하기. 안그러면 config.json 파일 못 읽음)

!python infer.py -t 0 <<< 0,1,2 중에 넣으면 캐릭터별로 음성 변조 됨