# ElegooSlicer MCP Server

Serveur **Model Context Protocol (MCP)** complet permettant à un assistant IA (Antigravity, Claude Desktop, Cursor, etc.) de piloter **ElegooSlicer** de bout en bout.

## Fonctionnalités couvertes

1. **Projets, Multi-plateaux & Interface Graphique**
   - `project_new`, `project_open`, `project_save`, `project_info`
   - `plate_list`, `plate_create`, `plate_delete`, `plate_move_object`, `plate_set_settings`
   - `gui_open_project` (envoie le projet `.3mf` directement dans l'instance ElegooSlicer ouverte via `WM_COPYDATA`)
   - `gui_sync_status`, `gui_sync_from_live`, `gui_sync_to_live` (synchronisation bidirectionnelle avec l'instance ElegooSlicer en cours)
   - `gui_live_ipc_call` (appel direct sur le serveur IPC TCP d'ElegooSlicer : `printer.*`, `user.*`, et `slicer.*` sur une build issue de cette branche)
2. **Aperçus visuels PNG (MCP ImageContent)**
   - `preview_plate` : rendu isométrique Z-buffer du plateau avec étiquettes objets et dimensions
   - `preview_object_faces` : planche 3 angles annotée (`F0`, `F1`, ...) pour visualiser les faces imprimables candidates
3. **Objets 3D, Rotation, Modifiers, Découpe planaire & Face imprimable**
   - `object_add` : import simple ou multiple (`.stl`, `.obj`, `.step`/`.stp`, `.3mf`) ou primitives (`cube`, `cylinder`, `disc`, `sphere`) avec arrangement automatique
   - `object_list_faces` : détection des faces planes imprimables (aire en mm², normale, appartenance à l'enveloppe extérieure, ratio de porte-à-faux)
   - `object_lay_on_face` : poser un objet sur une face imprimable choisie (`face_id` ou normale)
   - `object_auto_orient`, `object_rotate`, `object_transform`, `object_arrange`, `object_remove`, `object_list`
   - `object_params_get`, `object_params_set` : paramètres spécifiques par objet (FR + EN)
   - `object_parts_list`, `object_modifier_add` : volumes modificateurs (`modifier_part`), volumes négatifs (`negative_part`), renforts/bloqueurs de supports (`support_enforcer`, `support_blocker`)
   - `object_cut` : coupe planaire en Z générant des demi-maillages fermés (`both`, `lower`, `upper`, option `flip_top`)
   - `object_layer_ranges_get`, `object_layer_ranges_set` : plages de hauteur de couche (modificateurs par intervalle Z)
4. **Contrôle du Filament & CANVAS / MMS**
   - `filament_list_profiles`, `filament_list_slots`, `filament_set_slots`, `filament_assign_to_object`
   - `filament_sync_from_printer` : lecture en direct des bobines du boîtier CANVAS/MMS sur l'imprimante connectée
5. **Tous les paramètres d'impression & profils (FR + EN + User Presets)**
   - `params_list_categories`, `params_search`, `params_get`, `params_set` (supporte les clés internes *et* les libellés français/anglais de l'UI comme `Hauteur de couche`, `Paroi extérieure`, `Position de la couture` = `Alignée`)
   - `profiles_list`, `profile_select` (résolution complète de l'héritage `"inherits"` sur les profils système et utilisateur)
   - `profile_save_user`, `profile_delete_user` : création, mise à jour et suppression persistante des profils utilisateur (`.json` + `.info`) dans `%APPDATA%\ElegooSlicer\user\default`
6. **Découpe, Export & Contrôle complet Imprimante (IPC + SDCP/RTM)**
   - `slice_project` : découpe headless via `elegoo-slicer.exe --slice` avec statistiques temps/filament/couches
   - `export_output` : export `.gcode`, `.gcode.3mf`, `.3mf`, `.stl`
   - `printer_list`, `printer_status`, `printer_mms_info`, `printer_files`, `printer_print`, `printer_upload_status`, `printer_upload_cancel`
   - `printer_tasks`, `printer_tasks_delete`, `printer_exceptions`, `printer_refresh_status`, `printer_pause`, `printer_resume`, `printer_stop`, `printer_rtm_command` (avec support `dry_run=true`)

## Démarrage rapide (zéro dépendance pip externe, Python 3.12 + NumPy)

```bash
python -m mcp_server
```

Ou en HTTP Streamable (`/mcp` POST, SSE GET, DELETE, `/health`, sessions `Mcp-Session-Id` et authentification Bearer optionnelle) :

```bash
python -m mcp_server --transport http --host 127.0.0.1 --port 8765 --auth-token mon-secret
```

Voir [`antigravity_mcp_config.example.json`](antigravity_mcp_config.example.json) pour l'ajout dans la configuration MCP de votre client.

## Tests E2E

```bash
python -m unittest discover -s mcp_server/tests -v
```
