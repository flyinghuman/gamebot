# Universal Game Automation Bot — Dokumentation (Deutsch)

Dieses Dokument beschreibt die Installation und den Betrieb des Bots, die Konfiguration eines Android/Windows-Backends (inkl. Bluestacks), die Bedienung der GUI sowie das Erstellen von Tasks. Ein vollständiges Beispiel basiert auf der `treasure_dig`-Aufgabe.

## 1. Voraussetzungen & schnelle Installation

Unterstützte Plattformen: Windows und Linux. Python 3.9+ wird benötigt.

Empfohlene Schnellinstallation (Linux / WSL / Windows PowerShell):

1. Klone das Repository
  ```bash
  git clone https://github.com/flyinghuman/gamebot.git
  cd gamebot
  ```

2. Erstelle und aktiviere eine virtuelle Umgebung:

   - Linux / macOS / WSL:

       python -m venv .venv
       source .venv/bin/activate

   - Windows (PowerShell):

       python -m venv .venv
       .\.venv\Scripts\Activate

3. Installiere die Python-Abhängigkeiten im Repository-Stamm:

       pip install -r requirements.txt

4. Konfiguriere ADB (falls Android-Backend) und wähle ein aktives Profil in `profiles/`.

Hinweise:
- Stelle sicher, dass `adb` (Android Platform Tools) im PATH ist, wenn das Android-Backend verwendet wird.
- Unter Windows kann es nötig sein, PowerShell oder die GUI als Administrator auszuführen.

## 2. Installation der Platform Tools / ADB

- Unter Linux (Debian/Ubuntu):

      sudo apt update
      sudo apt install adb

- Unter Windows: Lade die Android SDK Platform Tools von Google herunter und füge `adb.exe` dem PATH hinzu oder kopiere sie in einen bekannten Ordner. Teste mit:

      adb devices

Wenn du einen Emulator oder ein Gerät verbinden willst, vergewissere dich, dass es aufgelistet ist. Nutze die Skripte im Verzeichnis um die Geräte anzuzeigen.
- ADB für Windows kann hier geladen werden: https://dl.google.com/android/repository/platform-tools-latest-windows.zip

## 3. Bluestacks: Kurz-Anleitung (ADB aktivieren & Performance auf Maximum)

Bluestacks ist ein geläufiger Android-Emulator unter Windows. Der Bot kann über ADB mit dem Emulator kommunizieren. Kurzanleitung:

1. Bluestacks installieren.
2. Öffne Bluestacks Einstellungen → Erweitert (oder Engine / Performance je nach Version).
   - Setze Performance / CPU / Memory auf maximale Höchstkonfiguration.
   - Übernehme die Einstellungen und starte Bluestacks ggf. neu.
3. Aktiviere ADB: In den Bluestacks-Einstellungen (Erweitert / ADB) den ADB-Schalter aktivieren.
4. Verbindung vom Host herstellen (falls nötig): Einige Bluestacks-Versionen bieten eine TCP-ADB-Schnittstelle (z. B. 127.0.0.1:5555). `adb connect <adresse>` nutzen.
5. Mit `adb devices` prüfen, ob Bluestacks gelistet ist.

Hinweise:
- Falls Bluestacks ADB nicht standardmäßig bereitstellt, konsultiere die Bluestacks-Doku zur jeweiligen Version.
- Aktiviere Virtualisierung (VT-x/AMD-V) im BIOS/UEFI für beste Emulationsleistung.

## 4. Den Bot starten

Aus dem Repository-Stamm mit aktivierter virtueller Umgebung:

    python bot.py

Wähle das gewünschte Profil in den Einstellungen (oder passe `profiles/active_profile.txt`) und starte den Bot über die GUI.

## 5. GUI-Übersicht — Beschreibung der Tabs

Monitor:
- Zeigt den Live-Screenshot-Feed des Backends (ADB oder Fensteraufnahme).
- Zeigt Heatmaps und ROI-Overlay, um Template-Erkennungen zu überprüfen.

Dashboard:
- Laufzeitstatistiken und konfigurierbare Task-Metriken.
- Recent Actions zeigt die letzten Aktionen mit Zeitstempel und Ergebnis.

ROIs (Regions of Interest):
- Listet alle Regions of Interest auf
- Erstellen und bearbeiten neuer ROIs mit Live Screenshots und Auswahl des Bereichs

Templates (Templates):
- Verwalte Template-Bilder, die in `config.yaml` referenziert werden.
- Pro Vorlage sind bis zu 10 Bilder möglich.

Tasks (Tasks):
- Erstellen, bearbeiten und anordnen von Tasks in `tasks.yaml`.
- Jeder Task hat `id`, `name`, `trigger` und `steps`.

Settings (Settings):
- Konfiguration (thresholds, ROIs, GUI-Optionen, Template-Pfad und Backend-Auswahl).

Log (Log):
- Strukturierte Ausgaben zum Debuggen von Task-Ausführungen, Fehlern und Erkennungsproblemen.

## 6. Task-Aufbau & Entwurf von Tasks

Beschreibung der wichtigsten Felder: `id`, `name`, `description`, `enabled`, `trigger`, `steps`, `stats`.

Wichtige Step-Typen: `tap_template`, `wait_tap_template`, `sleep`, `loop`, `if`, `set_flag`, `set_detail`, `set_success`, `call_task`, `press_back`.

Tipps:
- Kleine, gut testbare Schritte sind robuster.
- `wait_tap_template` verwenden, um auf UI-Elemente zu warten.
- Wiederverwendbare Routinen mit `call_task` kapseln.

## 7. Statistiken & Zähler (Dashboard)

Tasks können eine `stats`-Sektion enthalten, die vom Dashboard gelesen wird. Beispiel:

```yaml
stats:
  counters:
    - name: treasures_found
      label: Gefundene Schätze
      source: action
      accumulate: sum
      unit: count

  series:
    - name: treasures_dug
      label: Gegrabene Schätze
      source: action
      max_points: 200
      unit: count
```

Die GUI aktualisiert diese Zähler, wenn der Task entsprechende Aktionen ausführt oder Flags setzt. Falls nichts angezeigt wird, Profil neu laden.

## 8. Beispiel: Treasure Hunt (`treasure_dig`) — Schritt-für-Schritt

Zweck: Öffnen der Schatz-Ansicht, Erkennen eines geteilten Dig-Messages, Durchführung des Digs und Einsammeln der Belohnung.

Haupt-Templates: `dig_excavator_info_button`, `dig_share_info_message`, `dig_button`, `dig_do_dig_button`, `dig_send_troops_button`, `dig_grab_gift_button`.

Vereinfachter Ablauf:
1. `tap_template`: `dig_excavator_info_button` — öffnet das Dig-UI.
2. `sleep`: 1–2s Wartezeit.
3. `tap_template`: `dig_share_info_message` Falls gefunden, werden Flags gesetzt.
4. `if` (Flag `dig_msg_found`): `wait_tap_template`-Reihe für `dig_button`, `dig_do_dig_button`, `dig_send_troops_button`, dann `set_flag`/`set_success` und `wait_tap_template` für `dig_grab_gift_button`.
5. `call_task`: `close_any_windows` — Schließt Dialoge und stabilisiert die UI.

## 9. Fehlerbehebung & Tipps

- Fehlende Templates: Prüfe die Bilder in `profiles/<name>/templates/` und nutze den Monitor was erkannt wird und was nicht.
- Probleme mit Windows-Fenstersteuerung: `control_mode` auf `windows` setzen und `windows_title` prüfen.
- ADB zeigt kein Gerät: `adb devices` testen und ggf. Treiber/Platform-tools prüfen.

## 10. Nächste Schritte

- Neue Templates über die Templates-Tab hinzufügen.
- Kleine Test-Tasks erstellen und `run_at_start` für schnelles Testen setzen.

---
Wenn du möchtest, unterstütze die Weiterentwicklung des Projekts mit einer kleinen Spende an den Entwickler. Danke!
```text
BTC: bc1qydkjt2gfxr7uz4pt4jldpzwxqty8nc76vtv9ss
ETH: 0x5D0F170eBc8caC2db4F9477E26A4858142abDEEB
XRP: rUvTGRaqg9Q7DTJoMRGRVBZ1fpKaZQcx5a
DOGE: DRPr2oh2ynd4FJ66BrPFfskE1jpEWWnBpm
DASH: Xnzed2r417hGfgLG4DqCCTSNPci6FbvgHW
```
