#ifndef SLIC3R_UTILS_ELEGOO_SLICER_IPC_HPP
#define SLIC3R_UTILS_ELEGOO_SLICER_IPC_HPP

#include "slic3r/Utils/IPCMessage.hpp"

namespace Slic3r {

// Handles the "slicer.*" IPC namespace inside IPCServer.
// Calls are marshaled onto the wxWidgets UI thread safely because Plater,
// PresetBundle, Model, and GLCanvas3D are not thread-safe.
class SlicerIPC
{
public:
    static IPCResponse handleRequest(const IPCRequest &request);
};

} // namespace Slic3r

#endif // SLIC3R_UTILS_ELEGOO_SLICER_IPC_HPP
