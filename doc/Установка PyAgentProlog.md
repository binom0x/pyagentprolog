## 0. Сборка пакета (`.tar.gz` и `.whl`)

Сначала установите `build`:
```bash
python -m pip install --upgrade build
```
Затем в корне проекта (где лежит `pyproject.toml`):
```bash
python -m build
```

Появятся два файла в `dist/`:

- `pyagentprolog-5.9.0.tar.gz` (исходники)
    
- `pyagentprolog-5.9.0-py3-none-any.whl` (колесо)

## 1. Варианты установки

### Глобально в текущее виртуальное окружение

```powershell
# из wheel (быстрее — не пересобирает)
python -m pip install dist/pyagentprolog-5.9.0-py3-none-any.whl

# или из tar.gz
python -m pip install dist/pyagentprolog-5.9.0.tar.gz
```

### С опциональными зависимостями

```powershell
# только ядро (numpy подтянется автоматически)
python -m pip install dist/pyagentprolog-5.9.0-py3-none-any.whl

# с Numba-ускорением арифметики
python -m pip install "dist/pyagentprolog-5.9.0-py3-none-any.whl[numba]"

# со всем сразу
python -m pip install "dist/pyagentprolog-5.9.0-py3-none-any.whl[all]"
```

> На Windows PowerShell квадратные скобки `[numba]` нужно **брать в кавычки** — иначе PowerShell интерпретирует их как wildcard. В cmd.exe и bash кавычки не нужны.

### В указанное пользователем место (`--target`)

```powershell
python -m pip install --target D:\myproj\libs dist/pyagentprolog-5.9.0-py3-none-any.whl
```

После этого папка `D:\myproj\libs` содержит подпапки `prolog/` и `pyagentprolog-5.9.0.dist-info/`. Чтобы Python находил пакет:

**Способ 1 — переменная окружения:**

```powershell
# Windows (текущая сессия)
$env:PYTHONPATH = "D:\myproj\libs"

# Windows (постоянно)
setx PYTHONPATH "D:\myproj\libs"
```

```bash
# Linux/macOS
export PYTHONPATH="/путь/к/libs:$PYTHONPATH"
```

**Способ 2 — в самом скрипте:**

```python
import sys
sys.path.insert(0, r"D:\myproj\libs")
import prolog
```

**Способ 3 — через `.pth`-файл в site-packages** (постоянно, без правок скриптов):

Создайте файл `D:\...\site-packages\myproj.pth` с одной строкой:

```
D:\myproj\libs
```

Python автоматически добавит эту папку в `sys.path` при старте.

## 2. Режим разработки (без пересборки)

Если будете править код и не хотите каждый раз пересобирать wheel:

```powershell
python -m pip install -e .
```

Editable-установка добавляет `src/` в путь импорта напрямую. Изменения в `.py` видны сразу. `src/pyagentprolog.egg-info/` — это её артефакт, его можно спокойно закоммитить в `.gitignore`.

## 3. Проверка после установки

```python
import prolog
print(prolog.__version__)   # 5.9.0

# Ядро
s = prolog.Solver("родитель(иван, алексей).")
print(s.find_solutions("родитель(иван, X)."))

# Агент
a = prolog.Agent("run :- write('ok'), nl.")
print(a.query_one("run."))

# Опциональные (если ставили с [numba] / [all])
print(prolog.arith_numba_status())

# Опциональные (если ставили с [all] и есть expert.py)
try:
    from prolog import Expert
    print("Expert доступен")
except ImportError as e:
    print("Expert не установлен:", e)
```

## 4. `.gitignore` (чтобы не коммитить артефакты сборки)

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.venv/
venv/
.pytest_cache/
*.spec
```

