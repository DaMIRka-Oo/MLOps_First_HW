# MLOps - First HW - Suleev Damir

## Введение

Приветсвую Глеба Борисенко и других случайно забредших.
Здесь расскажу о имеющихся в проекте функциях и о том, как они работают.

### 1. /feature_description
Показывает доступные датасеты и модели, а также их параметры. 
В случае POST-запроса обязательно указывать ключ **'data_type'**

Для того, чтобы увидеть весь список доступных видов датасетов и моделей,
необходимо сделать GET-запрос

Чтобы увидеть описание определенного датасета, нужно задать значение ключа
равным **'Datasets'** и добавить дополнительную пару ключ-значение
**'dataset': 'dataset_name'**

Чтобы увидеть доступные гиперпараметры определенной модели, нужно задать
значение ключа равным **'Models'** и добавить дополнительную пару ключ-значение
**'model': 'model_name'**

### 2. /train_model
Функция, обучающая модели. 

Обязательно нужно задать, на каком датасете ты хочешь учиться
**dataset_nm** и тип обучаемой модели **model_type**. 

Можно задать название модели **model_nm** самому, но если данный ключ 
не задавать, то название подберется автоматически. Нельзя подавать
уже существующие названия моделей.

Список доступных гиперпараметров для каждого вида моделей можно
посмотреть в функции **feature_description**.

Разбиение выборки на *train* и *test* производится с параметрами 
**test_size=0.33** и **random_state=666**. При желании, эти параметры
также можно задать в виде пары ключ-значение.

### 3. /retraining
Переобучает существующую модель.

Все параметры совпадают с параметрами предыдущей функции.

Обязательно задавать имя существующей модели **model_nm**,
датасет **dataset_nm** и тип модели **model_nm**.

### 4. /remove_models
Удаляет существующую модель или все имеющиеся модели.

Обязательно задавать ключ **remove_list**.

Чтобы удалить все модели, нужно задать значение *All*.

Что удалить только нужную часть моделей, необходимо перечислить
их названия в списке

### 5. /show_models
Показывает список всех доступных моделей. 
Достаточно отправить GET-запрос

Нужно отправить сюда следующий файл json: **{'models_list': 'All'}**

### 6. /predict_class
Предсказыват класс наблюдения/наблюдений.

Необходимо подать имя модели **"model_nm"** и данные **"data"**
в виде пар ключ-значение для одного наблюдения, или списка таких пар для несколких наблюдений

Например:
#### Одно наблюдение
"data": {"alcohol": 14.23,
        "malic_acid": 1.71,
        "ash": 2.43,
        "alcalinity_of_ash": 15.6,
        "magnesium": 127.0,
        "total_phenols": 2.8,
        "flavanoids": 3.06,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 2.29,
        "color_intensity": 5.64,
        "hue": 1.04,
        "od280/od315_of_diluted_wines": 3.92,
        "proline": 1065.0}

#### Несколько наблюдений
"data": [{"alcohol": 14.23,
        "malic_acid": 1.71,
        "ash": 2.43,
        "alcalinity_of_ash": 15.6,
        "magnesium": 127.0,
        "total_phenols": 2.8,
        "flavanoids": 3.06,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 2.29,
        "color_intensity": 5.64,
        "hue": 1.04,
        "od280/od315_of_diluted_wines": 3.92,
        "proline": 1065.0},
    {"alcohol": 14.23,
        "malic_acid": 1.71,
        "ash": 2.43,
        "alcalinity_of_ash": 15.6,
        "magnesium": 127.0,
        "total_phenols": 2.8,
        "flavanoids": 3.06,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 2.29,
        "color_intensity": 5.64,
        "hue": 1.04,
        "od280/od315_of_diluted_wines": 3.92,
        "proline": 1065.0}]

Класс определеляется на основе **cutoff**, который по умолчанию
равен 0.5. Это параметр также можно задать самостоятельно


# Second HW

## Docker

Образ лежит на DockerHub под названием **dsuleev/mlops_suleev:v1**

Логин: **dsuleev**