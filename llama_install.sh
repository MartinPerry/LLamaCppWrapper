#!/bin/bash

BASE_PATH="e:/Programming/Cpp/llama.cpp/build"

cd ${BASE_PATH}
cd ..

git reset --hard
git pull

#========================

cmake -B "${BASE_PATH}" -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build "${BASE_PATH}" --config Release -j 8
cmake --build "${BASE_PATH}" --config Debug -j 8

#==================================================================

SOURCE1_DEBUG="${BASE_PATH}/ggml/src/Debug"
SOURCE2_DEBUG="${BASE_PATH}/common/Debug"
SOURCE3_DEBUG="${BASE_PATH}/src/Debug"
SOURCE4_DEBUG="${BASE_PATH}/tools/mtmd/Debug"
DEST_DEBUG="${BASE_PATH}/bin/Debug"

SOURCE1_RELEASE="${BASE_PATH}/ggml/src/Release"
SOURCE2_RELEASE="${BASE_PATH}/common/Release"
SOURCE3_RELEASE="${BASE_PATH}/src/Release"
SOURCE4_RELEASE="${BASE_PATH}/tools/mtmd/Release"
DEST_RELEASE="${BASE_PATH}/bin/Release"

# Function to copy .lib files
copy_libs() {
    local src1="$1"
    local src2="$2"
    local src3="$3"
	local src4="$4"
    local dest="$5"
    local type="$6"

    for src in "${src1}" "${src2}" "${src3}" "${src4}"; do
        if [ ! -d "${src}" ]; then
            echo "Source directory does not exist: ${src}"
            exit 1
        fi
    done

    mkdir -p "${dest}"

    cp -u "${src1}"/*.lib "${dest}" 2>/dev/null
    cp -u "${src2}"/*.lib "${dest}" 2>/dev/null
    cp -u "${src3}"/*.lib "${dest}" 2>/dev/null
	cp -u "${src4}"/*.lib "${dest}" 2>/dev/null

    echo "${type} files copied successfully!"
}

copy_libs "${SOURCE1_DEBUG}" "${SOURCE2_DEBUG}" "${SOURCE3_DEBUG}" "${SOURCE4_DEBUG}" "${DEST_DEBUG}" "Debug"
copy_libs "${SOURCE1_RELEASE}" "${SOURCE2_RELEASE}" "${SOURCE3_RELEASE}" "${SOURCE4_RELEASE}" "${DEST_RELEASE}" "Release"

#extension header copy to include dir
# this is a staging header for new llama.cpp API
# breaking changes and C++ are allowed. everything here should be considered WIP
# try as much as possible to not include this header in the rest of the codebase
cp -u "${BASE_PATH}/../src/llama-ext.h" "${BASE_PATH}/../include/llama-ext.h"