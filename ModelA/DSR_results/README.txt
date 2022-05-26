První model identifikující, zdali se na obrázku vyskytuje želva

Seznam souborů
final_alpha.h5 -> první model A, plně užitelný, nemožný replikace
final_beta.h5 -> první model A, plně užitelný, možný replikace, avšak nevyhovující po ručním testování
final_retrained_a.h5 -> model A retrénován na základě upravených datech
final_retrained_b.h5 -> model A retrénován s +1 hloubkovou vrstvou na základě nových datech
model_A.py -> první skript, čistě testovací, nepoužívat
model_A_final.py -> skript jehož výsledkem je generování souboru final_beta.h5
model_A_final_after_test_a.py -> skript jehož výsledkem je final_retrained_a.h5
model_A_final_after_test_b.py -> skript jehož výsledkem je final_retrained_b.h5
model_A_test_personal.py -> zjednodušující skript model_A_final.py, pomocí tohoto skriptu lze ručně testovat jednotlivé obrázky
model_A_lib_import.py -> skript obsahující funkci importování všech potřebných balíčků

Pro otestování
Užít python file model_A_test_personal.py, nutnost mít knihovny model_A_lib_import.py
1) final_retrained_a.h5 -> načíst z gitu
2) final_retrained_b.h5 -> stáhnout z uložta