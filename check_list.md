Hier sind die Regeln, abgeleitet aus dem Dokument, als Best Practices für Python-Blender-Add-ons:

Keine persistenten Referenzen auf Blender-Daten halten
Halte keine direkten Referenzen auf Object, Mesh, Bone, Collection-Items, Vertices, Polygons, Constraints, Modifier oder andere Blender-Daten, wenn deren Container verändert werden kann oder Undo/Redo möglich ist. Speichere stattdessen Indizes, Namen nur kontrolliert, oder eigene Mapping-Strukturen.
Nach Container-Änderungen Daten immer neu holen
Sobald du Collections erweiterst, Punkte hinzufügst, Mesh-Daten veränderst oder sonstige RNA-Container modifizierst, sind alte Referenzen potenziell ungültig. Greife danach immer erneut auf die Daten zu.
Undo/Redo als vollständige Invalidierung behandeln
Gehe grundsätzlich davon aus, dass Undo und Redo alle bpy.types.ID-Instanzen und deren Unterdaten ungültig machen. Add-ons dürfen sich nie darauf verlassen, dass alte Pointer nach einem Undo noch gültig sind.
Jede Datenänderung undo-sicher machen
Wenn ein Add-on Blender-Daten verändert, muss dafür ein sauberer Undo-Schritt existieren. Besonders Operatoren dürfen keine stillen Änderungen am Datenmodell durchführen, die den Undo-Stack beschädigen.
Keine komplexe Logik in RNA-Property-Callbacks
Update-Callbacks von Properties müssen klein, schnell und deterministisch bleiben. Keine Operator-Aufrufe, keine komplexen Seiteneffekte, keine rekursiven Änderungsketten.
Nach Mode-Wechsel alle Subdaten neu referenzieren
Nach Wechseln zwischen EDIT, OBJECT oder anderen Modi dürfen Referenzen auf Mesh-Polygone, Vertices, UVs, Bones, Curve-Points usw. nicht weiterverwendet werden. Nur das übergeordnete Datenobjekt kann erneut benutzt werden, Unterdaten müssen neu gelesen werden.
Geometrie nicht inkrementell über RNA aufbauen, wenn Reallokation droht
Beim Hinzufügen von Vertices, Curve-Points oder ähnlichen Elementen entstehen Reallokationen. Besser: Daten gesammelt anlegen oder nach jeder Erweiterung Referenzen erneuern. Für echte Geometriebearbeitung ist oft bmesh die richtige API.
Nach remove() nie mehr auf gelöschte Daten oder deren Subdaten zugreifen
Nicht nur das entfernte Objekt selbst ist tabu, sondern auch zuvor gespeicherte Unterreferenzen wie mesh.vertices. Diese können weiterhin auf freigegebenen Speicher zeigen.
Nie während Iterator-Läufen die zugrundeliegende Datenstruktur destabilisieren
Wenn Änderungen Caches, Sortierungen oder Sammlungen beeinflussen, zuerst eine stabile Kopie erzeugen, etwa mit [:], list(...) oder tuple(...). Dann iterieren. Das gilt besonders für Collection.all_objects und für Umbenennungen in bpy.data.*.
Keine Hintergrund-Threads mit Blender-API laufen lassen
Python-Threads sind für Blender-API-Zugriffe nicht threadsicher. Keine dauerhaften oder parallelen Threads, die bpy benutzen. Wenn Threads überhaupt genutzt werden, dann nur für externe Arbeit und nur solange der Hauptthread blockiert ist; vor erneutem bpy-Zugriff müssen alle Threads beendet sein. Für unabhängige Prozesse ist multiprocessing vorzuziehen.
Keine Blender-Python-Wrapper für Persistenz missbrauchen
Blender-Python-Objekte sind keine verlässlichen Träger für langlebigen Python-Zustand. Zusätzliche Zustände gehören in eigene Python-Strukturen, PropertyGroups oder sauber serialisierte Daten, nicht an zufällig existente Wrapper-Instanzen.
Objekte nicht blind über Namen wiederfinden
Namen in Blender sind nicht stabil genug als Primärschlüssel: sie können gekürzt, dedupliziert, kollidierend oder bibliotheksbezogen sein. Verwende eigene Mappings statt späterer Lookups über bpy.data[...], wenn du erzeugte Daten wiederfinden musst.
Nach Änderungen abhängige Daten explizit aktualisieren
Wenn du Eigenschaften setzt und sofort abhängige Werte brauchst, rufe bpy.context.view_layer.update() auf. Blender wertet vieles verzögert aus; ohne Update liest du oft veraltete Zustände.
UI-Kontextänderungen nicht als sofort wirksam annehmen
Änderungen an Workspace, Screen, Scene, Area oder ähnlichem sind oft verzögert. Logik, die den neuen Kontext benötigt, gehört in Timer, Handler oder Modal-Operatoren, nicht direkt in die unmittelbar folgende Zeile.
Lange Abläufe als Modal Operator bauen
Kein blockierender Python-Loop mit Redraw-Hacks. Für interaktive, wiederkehrende oder lange laufende Prozesse Modal-Operatoren verwenden. Das entspricht Blenders Architektur und reduziert Instabilität.
Operatoren nur verwenden, wenn sie wirklich passen
bpy.ops ist kontextabhängig und oft ungeeignet für robuste Add-on-Logik. Bevorzuge direkte API-Zugriffe (bpy.data, bmesh, RNA), wenn möglich. Operatoren nur dort einsetzen, wo der Kontext kontrolliert ist oder keine API-Alternative existiert.
Edit-Mode und Object-Mode bewusst trennen
Mesh-Daten in Edit-Mode und Object-Mode sind nicht automatisch synchron. Ein Add-on muss explizit entscheiden, ob es über obj.data, bmesh.from_edit_mesh() oder nach einem Mode-Wechsel arbeitet. Mischformen ohne klare Synchronisierung erzeugen Fehler.
Für Geometrie-Manipulation bmesh, für Export je nach Zielstruktur
Für Erzeugung und Bearbeitung von Mesh-Geometrie ist bmesh meist die beste Wahl. MeshPolygon ist eher Speicherformat, MeshLoopTriangle eher Exportformat für triangulierte Ziele. Nicht die falsche Datenstruktur für den falschen Zweck verwenden.
Armature-Datentypen strikt unterscheiden
edit_bones, bones und pose.bones sind nicht austauschbar. Jeder Typ gehört zu einem anderen Modus und Einsatzzweck. Mode-Wechsel dürfen keine alten Bone-Referenzen mitnehmen, besonders nicht aus Edit-Mode.
Dateipfade Blender-konform behandeln
Relative Pfade mit // immer über bpy.path.abspath() auflösen, bei verlinkten Libraries mit library=.... Pfade nicht roh als normale Strings behandeln. Für Encoding robust mit UTF-8, os.fsencode() und os.fsdecode() arbeiten.
Library-Daten nur mit voller Kontrolle verändern
Änderungen an verlinkten Library-Daten sind ein Sonderfall mit Undo-Risiken. Für normale Add-ons ist das zu vermeiden. Nur einsetzen, wenn die Lebenszyklen der referenzierten Daten vollständig kontrolliert werden.
Defensiv gegen Speicher- und Corner-Case-Probleme programmieren
Keine Rekursion, die Speicher- oder Referenzprobleme verschleiert. Große Listen sparsam behandeln. Keine Annahmen treffen, dass ein Crash reproduzierbar sein muss. Add-on-Code muss konservativ gebaut sein.

Kompaktform für Add-on-Standards:

Nie auf alte Blender-Referenzen vertrauen.
Direkte API vor Operatoren, Modal vor blockierend, bmesh vor RNA-Gefrickel bei Geometrie.

Quelle: