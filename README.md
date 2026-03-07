# Instrukcja 

## Konfiguracja interpretera Python, instalacja wymaganych modułów

Do uruchamiania skryptów i notatników można wykorzystać wirtualne środowisko .venv, które można utworzyć wewnątrz 
głównego katalogu:

```shell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Następnie należy pobrać wszystkie wymagane moduły zapisane w pliku requirements.txt:

```shell
pip install -r .\requirements.txt
```

## Dodanie zbiorów danych
W folderach Classification i Regression należy utworzyć foldery *data* i do nich dodać pliki .csv zawierające zbiory
danych, na których będą tworzone modele.
