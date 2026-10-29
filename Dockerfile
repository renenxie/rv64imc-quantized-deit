FROM ubuntu:24.04
ARG TARGETARCH
ARG XPACK_VER=15.2.0-1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates device-tree-compiler \
      python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

# RISC-V GCC（xPack 預編譯版，自動選 arm64 / x64）
RUN case "$TARGETARCH" in amd64) A=x64;; arm64) A=arm64;; esac \
    && curl -fsSL https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v${XPACK_VER}/xpack-riscv-none-elf-gcc-${XPACK_VER}-linux-${A}.tar.gz \
       | tar xz -C /opt \
    && ln -s /opt/xpack-riscv-none-elf-gcc-${XPACK_VER} /opt/xpack-gcc
ENV PATH=/opt/xpack-gcc/bin:/opt/riscv/bin:$PATH

# Spike（關掉 Boost 以避開編譯錯誤）
RUN git clone --depth 1 https://github.com/riscv-software-src/riscv-isa-sim.git /tmp/spike \
    && cd /tmp/spike && mkdir build && cd build \
    && ../configure --prefix=/opt/riscv --without-boost --without-boost-asio --without-boost-regex \
    && make -j$(nproc) && make install && rm -rf /tmp/spike

# pk（proxy kernel），並建 symlink 讓 `spike pk` 找得到
RUN git clone --depth 1 https://github.com/riscv-software-src/riscv-pk.git /tmp/pk \
    && cd /tmp/pk && mkdir build && cd build \
    && ../configure --prefix=/opt/riscv --host=riscv-none-elf --with-arch=rv64imac_zicsr_zifencei \
    && make -j$(nproc) && make install && rm -rf /tmp/pk \
    && ln -s /opt/riscv/riscv-none-elf /opt/riscv/riscv64-unknown-elf

WORKDIR /work
