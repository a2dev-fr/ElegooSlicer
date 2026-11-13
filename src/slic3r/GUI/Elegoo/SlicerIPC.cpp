#include "slic3r/Utils/Elegoo/SlicerIPC.hpp"
#include "slic3r/Utils/Elegoo/IPCCommon.hpp"
#include "slic3r/GUI/GUI_App.hpp"
#include "slic3r/GUI/MainFrame.hpp"
#include "slic3r/GUI/Plater.hpp"
#include "slic3r/GUI/Selection.hpp"
#include "slic3r/GUI/GLCanvas3D.hpp"
#include "libslic3r/AppConfig.hpp"
#include "libslic3r/Config.hpp"
#include "libslic3r/Format/bbs_3mf.hpp"
#include "libslic3r/Geometry.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "libslic3r/PrinterNetworkInfo.hpp"

#include <wx/utils.h>
#include <boost/filesystem.hpp>
#include <cmath>
#include <future>
#include <string>
#include <vector>

namespace Slic3r {

namespace {

constexpr double PI_VAL = 3.14159265358979323846;

inline double deg2rad(double deg) { return deg * (PI_VAL / 180.0); }
inline double rad2deg(double rad) { return rad * (180.0 / PI_VAL); }

nlohmann::json serialize_object_info(const ModelObject *obj, size_t idx)
{
    nlohmann::json j;
    j["index"] = idx;
    j["id"]    = obj->id().id;
    j["name"]  = obj->name;

    std::string extruder_val = "0";
    if (const auto *opt = obj->config.option("extruder"))
        extruder_val = opt->serialize();
    j["extruder"] = extruder_val;

    const BoundingBoxf3 bbox = obj->raw_mesh_bounding_box();
    j["raw_size_mm"] = {bbox.size().x(), bbox.size().y(), bbox.size().z()};

    nlohmann::json instances_json = nlohmann::json::array();
    for (size_t i = 0; i < obj->instances.size(); ++i) {
        const ModelInstance *inst = obj->instances[i];
        const Vec3d off   = inst->get_offset();
        const Vec3d rot   = inst->get_rotation();
        const Vec3d scale = inst->get_scaling_factor();
        const Vec3d mir   = inst->get_mirror();
        nlohmann::json ij;
        ij["index"]      = i;
        ij["offset_mm"]  = {off.x(), off.y(), off.z()};
        ij["rotation_deg"] = {rad2deg(rot.x()), rad2deg(rot.y()), rad2deg(rot.z())};
        ij["scale"]      = {scale.x(), scale.y(), scale.z()};
        ij["mirror"]     = {mir.x(), mir.y(), mir.z()};
        ij["printable"]  = inst->printable;
        instances_json.push_back(ij);
    }
    j["instances"] = instances_json;
    return j;
}

std::string get_config_option_serialized(const DynamicPrintConfig &cfg, const std::string &key)
{
    if (const ConfigOption *opt = cfg.option(key))
        return opt->serialize();
    return "";
}

IPCResponse handle_on_ui_thread(const IPCRequest &request)
{
    IPCResponse resp;
    resp.id     = request.id;
    resp.method = request.method;
    resp.code   = static_cast<int>(PrinterNetworkErrorCode::SUCCESS);

    const std::string dot = "slicer.";
    const std::string action =
        (request.method.rfind(dot, 0) == 0) ? request.method.substr(dot.size()) : request.method;

    GUI::GUI_App &app = GUI::wxGetApp();
    GUI::Plater *plater = app.plater();
    if (!plater) {
        resp.code    = static_cast<int>(PrinterNetworkErrorCode::OPERATION_FAILED);
        resp.message = "Plater is not initialized";
        return resp;
    }

    PresetBundle *presets = app.preset_bundle;
    const auto   &params  = request.params;

    if (action == "ping") {
        resp.data = {
            {"ok", true},
            {"objects_count", plater->model().objects.size()},
            {"has_presets", presets != nullptr}
        };
        return resp;
    }

    if (action == "getState") {
        nlohmann::json objs = nlohmann::json::array();
        for (size_t i = 0; i < plater->model().objects.size(); ++i)
            objs.push_back(serialize_object_info(plater->model().objects[i], i));

        nlohmann::json filaments = nlohmann::json::array();
        if (presets) {
            const auto colors = presets->project_config.option("filament_colour")
                ? presets->project_config.option("filament_colour")->serialize()
                : "";
            for (size_t i = 0; i < presets->filament_presets.size(); ++i) {
                filaments.push_back({
                    {"slot", i + 1},
                    {"preset", presets->filament_presets[i]}
                });
            }
            resp.data["filament_colours"] = colors;
            resp.data["print_preset"]     = presets->prints.get_selected_preset_name();
            resp.data["printer_preset"]   = presets->printers.get_selected_preset_name();
        }
        resp.data["objects"]   = objs;
        resp.data["filaments"] = filaments;
        return resp;
    }

    if (action == "newProject") {
        const std::string name = params.value("name", "");
        plater->new_project(true, true, wxString::FromUTF8(name));
        resp.data = {{"ok", true}};
        return resp;
    }

    if (action == "openProject") {
        const std::string path = params.value("path", "");
        if (path.empty() || !boost::filesystem::exists(path)) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Project file not found: " + path;
            return resp;
        }
        plater->load_project(wxString::FromUTF8(path));
        resp.data = {{"ok", true}, {"path", path}, {"objects_count", plater->model().objects.size()}};
        return resp;
    }

    if (action == "saveProject" || action == "export3mf") {
        const std::string path = params.value("path", "");
        if (path.empty()) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Missing 'path' parameter";
            return resp;
        }
        const int plate_idx = params.value("plate_index", -1);
        const int rc        = plater->export_3mf(boost::filesystem::path(path), SaveStrategy::Default, plate_idx);
        resp.data = {{"ok", rc == 0}, {"return_code", rc}, {"path", path}};
        return resp;
    }

    if (action == "addObject") {
        std::vector<std::string> paths;
        if (params.contains("paths") && params["paths"].is_array()) {
            for (const auto &p : params["paths"])
                if (p.is_string())
                    paths.push_back(p.get<std::string>());
        } else if (params.contains("path") && params["path"].is_string()) {
            paths.push_back(params["path"].get<std::string>());
        }
        if (paths.empty()) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Missing 'path' or 'paths'";
            return resp;
        }
        const auto loaded = plater->load_files(paths, LoadStrategy::LoadModel, false);
        resp.data = {
            {"ok", !loaded.empty()},
            {"loaded_indices", loaded},
            {"objects_count", plater->model().objects.size()}
        };
        return resp;
    }

    if (action == "removeObject") {
        if (params.value("all", false)) {
            plater->delete_all_objects_from_model();
            resp.data = {{"ok", true}, {"objects_count", plater->model().objects.size()}};
            return resp;
        }
        const int idx = params.value("object_index", -1);
        if (idx < 0 || static_cast<size_t>(idx) >= plater->model().objects.size()) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Invalid object_index";
            return resp;
        }
        const bool ok = plater->delete_object_from_model(static_cast<size_t>(idx), true);
        resp.data = {{"ok", ok}, {"objects_count", plater->model().objects.size()}};
        return resp;
    }

    if (action == "listObjects") {
        nlohmann::json objs = nlohmann::json::array();
        for (size_t i = 0; i < plater->model().objects.size(); ++i)
            objs.push_back(serialize_object_info(plater->model().objects[i], i));
        resp.data = {{"objects", objs}};
        return resp;
    }

    if (action == "rotateObject" || action == "transformObject") {
        const int obj_idx  = params.value("object_index", -1);
        const int inst_idx = params.value("instance_index", 0);
        if (obj_idx < 0 || static_cast<size_t>(obj_idx) >= plater->model().objects.size()) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Invalid object_index";
            return resp;
        }
        ModelObject *obj = plater->model().objects[obj_idx];
        if (inst_idx < 0 || static_cast<size_t>(inst_idx) >= obj->instances.size()) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Invalid instance_index";
            return resp;
        }
        ModelInstance *inst = obj->instances[inst_idx];
        const bool relative = params.value("relative", false);

        if (params.contains("rotation_deg") && params["rotation_deg"].is_array() && params["rotation_deg"].size() == 3) {
            Vec3d r(deg2rad(params["rotation_deg"][0].get<double>()),
                    deg2rad(params["rotation_deg"][1].get<double>()),
                    deg2rad(params["rotation_deg"][2].get<double>()));
            if (relative)
                r += inst->get_rotation();
            inst->set_rotation(r);
        }
        if (params.contains("offset_mm") && params["offset_mm"].is_array() && params["offset_mm"].size() == 3) {
            Vec3d o(params["offset_mm"][0].get<double>(),
                    params["offset_mm"][1].get<double>(),
                    params["offset_mm"][2].get<double>());
            if (relative)
                o += inst->get_offset();
            inst->set_offset(o);
        }
        if (params.contains("scale") && params["scale"].is_array() && params["scale"].size() == 3) {
            Vec3d s(params["scale"][0].get<double>(),
                    params["scale"][1].get<double>(),
                    params["scale"][2].get<double>());
            inst->set_scaling_factor(s);
        } else if (params.contains("scale") && params["scale"].is_number()) {
            const double s = params["scale"].get<double>();
            inst->set_scaling_factor(Vec3d(s, s, s));
        }
        if (params.contains("mirror") && params["mirror"].is_array() && params["mirror"].size() == 3) {
            Vec3d m(params["mirror"][0].get<double>(),
                    params["mirror"][1].get<double>(),
                    params["mirror"][2].get<double>());
            inst->set_mirror(m);
        }
        if (params.value("ensure_on_bed", true))
            obj->ensure_on_bed();
        plater->changed_object(*obj);
        resp.data = {{"ok", true}, {"object", serialize_object_info(obj, static_cast<size_t>(obj_idx))}};
        return resp;
    }

    if (action == "layOnFace") {
        const int obj_idx = params.value("object_index", -1);
        if (obj_idx < 0 || static_cast<size_t>(obj_idx) >= plater->model().objects.size()) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Invalid object_index";
            return resp;
        }
        if (!params.contains("normal") || !params["normal"].is_array() || params["normal"].size() != 3) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Missing normal [nx, ny, nz]";
            return resp;
        }
        ModelObject *obj = plater->model().objects[obj_idx];
        const int inst_idx = params.value("instance_index", 0);
        if (inst_idx < 0 || static_cast<size_t>(inst_idx) >= obj->instances.size()) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Invalid instance_index";
            return resp;
        }
        Vec3d n(params["normal"][0].get<double>(),
                params["normal"][1].get<double>(),
                params["normal"][2].get<double>());
        if (n.norm() < 1e-9) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Normal vector cannot be zero";
            return resp;
        }
        n.normalize();
        const Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(n, Vec3d(0.0, 0.0, -1.0));
        obj->instances[inst_idx]->rotate(q.toRotationMatrix());
        obj->ensure_on_bed();
        plater->changed_object(*obj);
        resp.data = {{"ok", true}, {"object", serialize_object_info(obj, static_cast<size_t>(obj_idx))}};
        return resp;
    }

    if (action == "arrangeObjects") {
        plater->arrange();
        resp.data = {{"ok", true}};
        return resp;
    }

    if (action == "autoOrientObject") {
        plater->orient();
        resp.data = {{"ok", true}};
        return resp;
    }

    if (action == "getFilaments") {
        nlohmann::json slots = nlohmann::json::array();
        if (presets) {
            for (size_t i = 0; i < presets->filament_presets.size(); ++i) {
                slots.push_back({{"slot", i + 1}, {"preset", presets->filament_presets[i]}});
            }
            resp.data = {
                {"slots", slots},
                {"filament_colour", get_config_option_serialized(presets->project_config, "filament_colour")},
                {"filament_type", presets->filaments.get_edited_preset().config.option("filament_type")
                    ? presets->filaments.get_edited_preset().config.option("filament_type")->serialize() : ""}
            };
        }
        return resp;
    }

    if (action == "addFilament") {
        plater->add_filament();
        resp.data = {{"ok", true}, {"count", presets ? presets->filament_presets.size() : 0}};
        return resp;
    }

    if (action == "removeFilament") {
        const int slot = params.value("slot", 0);
        if (slot >= 1)
            plater->delete_filament(static_cast<size_t>(slot - 1));
        else
            plater->delete_filament();
        resp.data = {{"ok", true}, {"count", presets ? presets->filament_presets.size() : 0}};
        return resp;
    }

    if (action == "setFilament") {
        if (!presets) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::OPERATION_FAILED);
            resp.message = "PresetBundle unavailable";
            return resp;
        }
        const int slot = params.value("slot", 1);
        if (slot < 1) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "slot must be >= 1";
            return resp;
        }
        while (presets->filament_presets.size() < static_cast<size_t>(slot))
            plater->add_filament();

        if (params.contains("preset") && params["preset"].is_string()) {
            presets->set_filament_preset(static_cast<size_t>(slot - 1), params["preset"].get<std::string>());
        }
        if (params.contains("color") && params["color"].is_string()) {
            if (auto *opt = presets->project_config.opt<ConfigOptionStrings>("filament_colour", true)) {
                while (opt->values.size() < presets->filament_presets.size())
                    opt->values.push_back("#FFFFFF");
                opt->values[static_cast<size_t>(slot - 1)] = params["color"].get<std::string>();
            }
        }
        app.load_current_presets(false, true);
        plater->update();
        resp.data = {{"ok", true}, {"slot", slot}, {"preset", presets->filament_presets[slot - 1]}};
        return resp;
    }

    if (action == "assignObjectFilament") {
        const int obj_idx = params.value("object_index", -1);
        const int slot    = params.value("slot", 1);
        if (obj_idx < 0 || static_cast<size_t>(obj_idx) >= plater->model().objects.size() || slot < 1) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Invalid object_index or slot";
            return resp;
        }
        ModelObject *obj = plater->model().objects[obj_idx];
        obj->config.set_key_value("extruder", new ConfigOptionInt(slot));
        plater->changed_object(*obj);
        resp.data = {{"ok", true}, {"object_index", obj_idx}, {"slot", slot}};
        return resp;
    }

    if (action == "getConfig") {
        if (!presets) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::OPERATION_FAILED);
            resp.message = "PresetBundle unavailable";
            return resp;
        }
        const std::string scope = params.value("scope", "print");
        const DynamicPrintConfig *target = &presets->prints.get_edited_preset().config;
        if (scope == "printer")
            target = &presets->printers.get_edited_preset().config;
        else if (scope == "filament")
            target = &presets->filaments.get_edited_preset().config;
        else if (scope == "project")
            target = &presets->project_config;

        nlohmann::json result = nlohmann::json::object();
        if (params.contains("keys") && params["keys"].is_array()) {
            for (const auto &k : params["keys"]) {
                if (k.is_string()) {
                    const std::string key = k.get<std::string>();
                    if (const ConfigOption *opt = target->option(key))
                        result[key] = opt->serialize();
                }
            }
        } else {
            for (const std::string &key : target->keys()) {
                if (const ConfigOption *opt = target->option(key))
                    result[key] = opt->serialize();
            }
        }
        resp.data = {{"scope", scope}, {"values", result}};
        return resp;
    }

    if (action == "setConfig") {
        if (!presets) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::OPERATION_FAILED);
            resp.message = "PresetBundle unavailable";
            return resp;
        }
        const std::string scope = params.value("scope", "print");
        DynamicPrintConfig *target = &presets->prints.get_edited_preset().config;
        if (scope == "printer")
            target = &presets->printers.get_edited_preset().config;
        else if (scope == "filament")
            target = &presets->filaments.get_edited_preset().config;
        else if (scope == "project")
            target = &presets->project_config;

        if (!params.contains("values") || !params["values"].is_object()) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::INVALID_PARAMETER);
            resp.message = "Missing 'values' object";
            return resp;
        }

        ConfigSubstitutionContext subs{ ForwardCompatibilitySubstitutionRule::Enable };
        nlohmann::json applied = nlohmann::json::array();
        nlohmann::json failed  = nlohmann::json::array();

        for (auto it = params["values"].begin(); it != params["values"].end(); ++it) {
            const std::string key = it.key();
            std::string val_str;
            if (it.value().is_string())
                val_str = it.value().get<std::string>();
            else if (it.value().is_boolean())
                val_str = it.value().get<bool>() ? "1" : "0";
            else
                val_str = it.value().dump();

            if (target->set_deserialize_nothrow(key, val_str, subs, false))
                applied.push_back(key);
            else
                failed.push_back(key);
        }

        plater->on_config_change(*target);
        app.load_current_presets(false, true);
        plater->update();
        resp.data = {{"ok", failed.empty()}, {"applied", applied}, {"failed", failed}};
        return resp;
    }

    if (action == "selectPreset") {
        if (!presets) {
            resp.code    = static_cast<int>(PrinterNetworkErrorCode::OPERATION_FAILED);
            resp.message = "PresetBundle unavailable";
            return resp;
        }
        const std::string kind = params.value("kind", "print");
        const std::string name = params.value("name", "");
        bool ok = false;
        if (kind == "print")
            ok = presets->prints.select_preset_by_name(name, true);
        else if (kind == "printer")
            ok = presets->printers.select_preset_by_name(name, true);
        else if (kind == "filament") {
            const int slot = params.value("slot", 1);
            presets->set_filament_preset(static_cast<size_t>(std::max(0, slot - 1)), name);
            ok = true;
        }
        app.load_current_presets(false, true);
        plater->update();
        resp.data = {{"ok", ok}, {"kind", kind}, {"name", name}};
        return resp;
    }

    if (action == "slice") {
        plater->reslice();
        resp.data = {{"ok", true}};
        return resp;
    }

    if (action == "exportGcode") {
        plater->export_gcode(false);
        resp.data = {{"ok", true}};
        return resp;
    }

    if (action == "exportGcode3mf") {
        const bool all_plates = params.value("all_plates", false);
        plater->export_gcode_3mf(all_plates);
        resp.data = {{"ok", true}};
        return resp;
    }

    resp.code    = static_cast<int>(PrinterNetworkErrorCode::OPERATION_NOT_IMPLEMENTED);
    resp.message = "Unknown slicer IPC method: " + request.method;
    return resp;
}

} // namespace

IPCResponse SlicerIPC::handleRequest(const IPCRequest &request)
{
    if (wxIsMainThread()) {
        return handle_on_ui_thread(request);
    }

    auto promise = std::make_shared<std::promise<IPCResponse>>();
    std::future<IPCResponse> future = promise->get_future();

    GUI::wxGetApp().CallAfter([request, promise]() {
        try {
            promise->set_value(handle_on_ui_thread(request));
        } catch (const std::exception &e) {
            IPCResponse err;
            err.id      = request.id;
            err.method  = request.method;
            err.code    = static_cast<int>(PrinterNetworkErrorCode::OPERATION_FAILED);
            err.message = e.what();
            promise->set_value(err);
        } catch (...) {
            IPCResponse err;
            err.id      = request.id;
            err.method  = request.method;
            err.code    = static_cast<int>(PrinterNetworkErrorCode::OPERATION_FAILED);
            err.message = "Unknown exception inside SlicerIPC";
            promise->set_value(err);
        }
    });

    return future.get();
}

} // namespace Slic3r
