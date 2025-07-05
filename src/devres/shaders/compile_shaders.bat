set VULKAN_SDK_DIR=C:/VulkanSDK/1.4.304.1/Bin
set OUTPUT_DIR=../../../res/shaders

%VULKAN_SDK_DIR%/glslangValidator.exe -V shader.vert -o %OUTPUT_DIR%/vert.spv
%VULKAN_SDK_DIR%/glslangValidator.exe -V shader.frag -o %OUTPUT_DIR%/frag.spv
pause