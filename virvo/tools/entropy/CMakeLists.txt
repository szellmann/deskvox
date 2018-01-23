find_package(Boost COMPONENTS filesystem serialization system REQUIRED)
deskvox_use_package(Boost)

deskvox_link_libraries(virvo)
deskvox_link_libraries(virvo_fileio)

deskvox_add_tool(entropy
  entropy.cpp
)
