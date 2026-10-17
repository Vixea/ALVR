use std::env;

fn main() {
    let lib_name = alvr_filesystem::dynlib_fname("alvr_client_openxr");

    let exe_adjacent_path = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|dir| dir.join(&lib_name)));

    unsafe {
        let lib = exe_adjacent_path
            .as_ref()
            .and_then(|path| libloading::Library::new(path).ok())
            .or_else(|| libloading::Library::new(&lib_name).ok())
            .unwrap_or_else(|| {
                panic!(
                    "Failed to load {lib_name}. Make sure it is in the same directory as the executable or in LD_LIBRARY_PATH."
                )
            });

        let entry_point: libloading::Symbol<unsafe extern "C" fn()> = lib
            .get(b"alvr_entry_point\0")
            .expect("Symbol alvr_entry_point not found in libalvr_client_openxr");

        entry_point();
    }
}
