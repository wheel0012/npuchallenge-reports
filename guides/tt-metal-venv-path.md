# 임의의 홈 디렉터리에서 `.venv`와 `tt-metal` 연결하기

이 문서는 사용자 이름이나 저장소 디렉터리 이름에 관계없이, 원하는 `tt-metal` 소스 경로를 Python 가상환경 `.venv`에서 사용하도록 설정하는 방법을 설명합니다.

예를 들어 다음과 같은 경로를 모두 같은 방식으로 설정할 수 있습니다.

```text
/home/alice/tt-metal
/home/bob/work/tt-metal-custom
/data/users/carol/tt-metal
```

## 1. 경로 지정

아래 두 값만 자신의 환경에 맞게 수정합니다.

```bash
TT_METAL_ROOT="/home/<사용자명>/<tt-metal 디렉터리>"
VENV_ROOT="/home/<사용자명>/.venv"
```

예:

```bash
TT_METAL_ROOT="/home/alice/tt-metal"
VENV_ROOT="/home/alice/.venv"
```

경로가 실제로 존재하는지 확인합니다.

```bash
test -d "$TT_METAL_ROOT" && echo "tt-metal 경로 확인 완료"
test -x "$VENV_ROOT/bin/python" && echo "가상환경 확인 완료"
```

아직 가상환경이 없다면 생성합니다.

```bash
python3 -m venv "$VENV_ROOT"
```

## 2. 권장 방법: editable install

가상환경의 Python과 pip를 사용해 `ttnn`을 editable 모드로 설치합니다. 현재 `tt-metal` 저장소는 프로젝트 루트에 `pyproject.toml`과 `setup.py`가 있으므로 루트 경로를 설치합니다.

```bash
source "$VENV_ROOT/bin/activate"
python -m pip install -e "$TT_METAL_ROOT"
```

다른 버전이나 포크를 사용하는 경우에는 `pyproject.toml` 또는 `setup.py`가 있는 위치로 판단할 수 있습니다.

```bash
find "$TT_METAL_ROOT" -maxdepth 2 \
  \( -name pyproject.toml -o -name setup.py \) -print
```

editable install을 사용하면 소스 코드를 수정했을 때 패키지를 매번 복사하거나 재설치하지 않아도 변경 내용이 반영됩니다.

## 3. 추가 경로가 필요할 때: `.pth` 파일 설정

`ttnn`, `tools`, 저장소 루트를 모두 Python 검색 경로에 포함해야 한다면 `.pth` 파일을 추가합니다.

먼저 현재 가상환경의 `site-packages` 위치를 구합니다.

```bash
SITE_PACKAGES="$("$VENV_ROOT/bin/python" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$SITE_PACKAGES"
```

그다음 `ttnn-custom.pth` 파일을 만듭니다.

```bash
printf '%s\n' \
  "$TT_METAL_ROOT" \
  "$TT_METAL_ROOT/ttnn" \
  "$TT_METAL_ROOT/tools" \
  > "$SITE_PACKAGES/ttnn-custom.pth"
```

생성된 내용을 확인합니다.

```bash
cat "$SITE_PACKAGES/ttnn-custom.pth"
```

예를 들어 `TT_METAL_ROOT=/home/alice/tt-metal`이면 다음 내용이 저장됩니다.

```text
/home/alice/tt-metal
/home/alice/tt-metal/ttnn
/home/alice/tt-metal/tools
```

> `.pth` 파일에는 `~`나 `$HOME` 대신 완전한 절대 경로를 기록하는 것이 안전합니다. `.pth` 파일 안에서는 셸 변수 확장이 적용되지 않습니다.

## 4. 설정 검증

가상환경 Python이 실제로 어떤 파일을 불러오는지 확인합니다.

```bash
"$VENV_ROOT/bin/python" -c \
  'import ttnn; print(ttnn.__file__)'
```

출력 경로가 지정한 `TT_METAL_ROOT` 아래를 가리켜야 합니다.

Python 검색 경로 전체도 확인할 수 있습니다.

```bash
"$VENV_ROOT/bin/python" -c \
  'import sys; print("\n".join(sys.path))'
```

editable install 정보는 다음처럼 확인합니다.

```bash
"$VENV_ROOT/bin/python" -m pip show ttnn
```

또는:

```bash
find "$SITE_PACKAGES" -path '*ttnn*.dist-info/direct_url.json' \
  -exec cat {} \;
```

`direct_url.json`에 다음과 비슷한 값이 있으면 editable install이 연결된 것입니다.

```json
{
  "dir_info": {"editable": true},
  "url": "file:///home/alice/tt-metal"
}
```

## 5. 기존의 잘못된 경로 변경

먼저 현재 `.pth` 파일을 확인합니다.

```bash
cat "$SITE_PACKAGES/ttnn-custom.pth"
```

새 `TT_METAL_ROOT` 값을 지정한 뒤 3절의 `printf` 명령을 다시 실행하면 `.pth` 파일의 경로가 교체됩니다.

editable install 경로도 바뀌었다면 기존 패키지를 제거한 후 새 경로로 다시 설치합니다.

```bash
source "$VENV_ROOT/bin/activate"
python -m pip uninstall -y ttnn
python -m pip install -e "$TT_METAL_ROOT"
```

마지막으로 반드시 `ttnn.__file__`을 출력해 새 경로가 사용되는지 확인합니다.

## 주의 사항

- 반드시 가상환경의 Python과 pip를 사용합니다. 가장 확실한 형태는 `"$VENV_ROOT/bin/python" -m pip ...`입니다.
- 동일한 `ttnn`이 여러 위치에 설치되어 있으면 `sys.path`에서 먼저 발견된 패키지가 사용됩니다.
- 저장소를 이동하거나 이름을 바꾸면 editable install과 `.pth` 파일도 새 절대 경로로 갱신해야 합니다.
- `.venv` 디렉터리를 다른 사용자에게 그대로 복사하기보다는, 새 위치에서 가상환경을 만들고 위 설정을 다시 적용하는 편이 안전합니다.
