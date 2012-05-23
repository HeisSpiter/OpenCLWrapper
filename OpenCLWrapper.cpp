#include "OpenCL.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <iostream>

struct KernelDef {
    std::string    Name;
    std::string    File;
    cl_device_type Target;
};

///
/// \fn     PrintUsage
/// \param  ProgName Name of the executable being run
/// \return 0
/// \brief  This function displays the information line about how to use the program
///
static int PrintUsage(const char * ProgName) {
    std::cout << ProgName << ": ConfigFile" << std::endl;
    return 0;
}

///
/// \fn     main
/// \param  argc Number of passed arguments (>= 1)
/// \param  argv All the passed arguments
/// \return 0 in case of success, -error otherwise
/// \brief  Main function
///
int main(int argc, char ** argv) {
    struct stat stbuf;
    xmlDocPtr XmlFile = 0;
    xmlNodePtr OldNode = 0;
    const char * ConfigFile;
    OpenCLWrapper::OpenCL OclObject;
    xmlXPathObjectPtr XmlObject = 0;
    xmlXPathContextPtr XmlContext = 0;
    KernelDef Kernel = {"", "", CL_DEVICE_TYPE_ALL};

    //
    // Check for the config file
    //
    if (argc != 2) {
        return PrintUsage(argv[0]);
    }

    ConfigFile = argv[1];

    //
    // Start config file parsing
    //
    XmlFile = xmlReadFile(ConfigFile, 0, 0);
    if (XmlFile == 0) {
        std::cerr << "Could not open: " << ConfigFile << std::endl;
        return -1;
    }

    XmlContext = xmlXPathNewContext(XmlFile);
    if (XmlContext == 0) {
        xmlFreeDoc(XmlFile);
        return -2;
    }

    //
    // First get the file name that contains the kernel to execute
    //
    XmlObject = xmlXPathEval(BAD_CAST"string(/kernel/@file)", XmlContext);
    if ((XmlObject != 0) && ((XmlObject->type == XPATH_STRING) &&
        (XmlObject->stringval != NULL) && (XmlObject->stringval[0] != 0))) {
        Kernel.File = reinterpret_cast<const char*>(XmlObject->stringval);
    }

    if (XmlObject) {
        xmlXPathFreeObject(XmlObject);
        XmlObject = 0;
    }

    //
    // Ensure that file name is correct and that the file exists
    //
    if (Kernel.File == "") {
        std::cout << "Kernel file name was not provided" << std::endl;
        xmlXPathFreeContext(XmlContext);
        xmlFreeDoc(XmlFile);
        return -3;
    }


    if (stat(Kernel.File.c_str(), &stbuf) != 0) {
        std::cout << "Kernel file was incorrect" << std::endl;
        xmlXPathFreeContext(XmlContext);
        xmlFreeDoc(XmlFile);
        return -3;
    }

    //
    // Get the kernel name
    //
    XmlObject = xmlXPathEval(BAD_CAST"string(/kernel/@name)", XmlContext);
    if ((XmlObject != 0) && ((XmlObject->type == XPATH_STRING) &&
        (XmlObject->stringval != NULL) && (XmlObject->stringval[0] != 0))) {
        Kernel.Name = reinterpret_cast<const char*>(XmlObject->stringval);
    }

    if (XmlObject) {
        xmlXPathFreeObject(XmlObject);
        XmlObject = 0;
    }

    //
    // Ensure that kernel name is not empty
    //
    if (Kernel.Name == "") {
        std::cout << "Kernel name was not provided" << std::endl;
        xmlXPathFreeContext(XmlContext);
        xmlFreeDoc(XmlFile);
        return -3;
    }

    //
    // Get target if any
    //
    XmlObject = xmlXPathEval(BAD_CAST"string(/kernel/target/@type)", XmlContext);
    if ((XmlObject != 0) && ((XmlObject->type == XPATH_STRING) &&
        (XmlObject->stringval != NULL) && (XmlObject->stringval[0] != 0))) {
        std::string Type = reinterpret_cast<const char*>(XmlObject->stringval);
        if (Type.compare("cpu") == 0) {
            Kernel.Target = CL_DEVICE_TYPE_CPU;
        } else if (Type.compare("gpu") == 0) {
            Kernel.Target = CL_DEVICE_TYPE_GPU;
        } else if (Type.compare("accelerator") == 0) {
            Kernel.Target = CL_DEVICE_TYPE_ACCELERATOR;
        }
    }

    //
    // Immediately set target to ensure it is well used
    //
    OclObject.SetParameter(OpenCLWrapper::TargetDevice, Kernel.Target);

    //
    // Unimplemented
    //
    assert(false);
    return 0;
}
