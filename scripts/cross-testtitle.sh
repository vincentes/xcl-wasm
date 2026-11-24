#!/bin/sh
# Builds XCL's test title (tests/testtitle, tests/testgamedll) as x64 Windows binaries on macOS or
# Linux: Rust's x86_64-pc-windows-gnu target with zig as the C compiler and the linker. The guest
# (crates/xcl-guest) runs the result; see crates/xcl-guest/tests/title.rs.
#
# Needs: zig, and  rustup toolchain install nightly --target x86_64-pc-windows-gnu
# Output: target/x86_64-pc-windows-gnu/debug/{xcl_testtitle.exe,testgame_mw.dll}
#
# This is not the build the Windows runtime is tested with (that one is MSVC, scripts/test.ps1).
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
tools="$root/target/cross-tools"
mkdir -p "$tools"

# the C compiler (build scripts): rustc's --target is not zig's
cat > "$tools/zcc" <<'EOF'
#!/bin/sh
for a in "$@"; do shift; case "$a" in --target=*) ;; *) set -- "$@" "$a";; esac; done
exec zig cc -target x86_64-windows-gnu "$@"
EOF

printf '#!/bin/sh\nexec zig ar "$@"\n' > "$tools/zar"
printf '#!/bin/sh\nexec zig dlltool "$@"\n' > "$tools/zdlltool"

# an import library zig does not ship: WaitOnAddress and friends (Rust's std links it)
printf 'LIBRARY api-ms-win-core-synch-l1-2-0.dll\nEXPORTS\nWaitOnAddress\nWakeByAddressSingle\nWakeByAddressAll\n' > "$tools/synchronization.def"
zig dlltool -m i386:x86-64 -d "$tools/synchronization.def" -l "$tools/libsynchronization.a"

# the linker: rustc's windows-gnu link line, for zig cc (which brings its own mingw CRT and
# libraries, so rustc's are dropped)
cat > "$tools/zld" <<EOF
#!/bin/sh
for a in "\$@"; do shift; case "\$a" in
  --target=*|-fno-use-linker-plugin|-Wl,--disable-auto-image-base|-nodefaultlibs|-lgcc_eh|-lgcc|-l:libpthread.a|-lmsvcrt|-lmingwex|-lmingw32|-nostartfiles) ;;
  -lsynchronization) set -- "\$@" "$tools/libsynchronization.a";;
  -Wl,*.def) set -- "\$@" "\${a#-Wl,}";;
  *crt2.o|*dllcrt2.o|*rsbegin.o|*rsend.o) ;;
  *) set -- "\$@" "\$a";;
esac; done
exec zig cc -target x86_64-windows-gnu "\$@" -lunwind
EOF

chmod +x "$tools/zcc" "$tools/zar" "$tools/zdlltool" "$tools/zld"

cd "$root"
CC_x86_64_pc_windows_gnu="$tools/zcc" \
AR_x86_64_pc_windows_gnu="$tools/zar" \
CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="$tools/zld" \
CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS="-Cdlltool=$tools/zdlltool" \
    cargo +nightly build -p xcl-testtitle -p testgame-mw --target x86_64-pc-windows-gnu "$@"

echo "built: $root/target/x86_64-pc-windows-gnu/debug/xcl_testtitle.exe"
