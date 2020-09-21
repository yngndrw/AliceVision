// This file is part of the AliceVision project.
// Copyright (c) 2020 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// Input and geometry
#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

// Image stuff
#include <aliceVision/image/all.hpp>
#include <aliceVision/mvsData/imageAlgo.hpp>

// Logging stuff
#include <aliceVision/system/Logger.hpp>

// Reading command line options
#include <boost/program_options.hpp>
#include <aliceVision/system/cmdline.hpp>
#include <aliceVision/system/main.hpp>

// IO
#include <fstream>
#include <algorithm>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>

#include <aliceVision/image/cache.hpp>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

using namespace aliceVision;

namespace po = boost::program_options;
namespace bpt = boost::property_tree;
namespace fs = boost::filesystem;

typedef struct {
  size_t offset_x;
  size_t offset_y;
  std::string img_path;
  std::string mask_path;
  std::string weights_path;
} ConfigView;

template <class T>
class CachedImage {
public:

  bool createImage(std::shared_ptr<image::TileCacheManager> manager, size_t width, size_t height) {

    _width = width;
    _height = height;
    _tileSize = manager->getTileWidth();

    _tilesArray.clear();

    int countHeight = int(ceil(double(height) / double(manager->getTileHeight())));
    int countWidth = int(ceil(double(width) / double(manager->getTileWidth())));

    _memoryWidth = countWidth * _tileSize;
    _memoryHeight = countHeight * _tileSize;

    for (int i = 0; i < countHeight; i++) {

      int tile_height = manager->getTileHeight();
      if (i == countHeight - 1) {
        tile_height = height - (i * tile_height);
      }

      std::vector<image::CachedTile::smart_pointer> row;

      for (int j = 0; j < countWidth; j++) {

        int tile_width = manager->getTileWidth();
        if (j == countWidth - 1) {
          tile_width = width - (i * tile_width);
        }

        image::CachedTile::smart_pointer tile = manager->requireNewCachedTile<T>(tile_width, tile_height);
        if (tile == nullptr) {
          return false;
        }

        row.push_back(tile);
      }

      _tilesArray.push_back(row);
    }

    return true;
  }

  bool writeImage(const std::string & path) {

    ALICEVISION_LOG_ERROR("incorrect template function");
    return false;
  }

  bool fill(const T & val) {

    for (int i = 0; i < _tilesArray.size(); i++) {

      std::vector<image::CachedTile::smart_pointer>& row = _tilesArray[i];

      for (int j = 0; j < _tilesArray[i].size(); j++) {

        image::CachedTile::smart_pointer ptr = row[j];
        if (!ptr) {
          continue;
        }

        if (!ptr->acquire()) {
          continue;
        }

        T * data = (T*)ptr->getDataPointer();

        std::fill(data, data + ptr->getTileWidth() * ptr->getTileHeight(), val);
      }
    }

    return true;
  }

  std::vector<std::vector<image::CachedTile::smart_pointer>> & getTiles() {
    return _tilesArray;
  }

private:
  size_t _width;
  size_t _height;
  size_t _memoryWidth;
  size_t _memoryHeight;
  size_t _tileSize;

  std::vector<std::vector<image::CachedTile::smart_pointer>> _tilesArray;
};

template <>
bool CachedImage<image::RGBAfColor>::writeImage(const std::string & path) {

  std::unique_ptr<oiio::ImageOutput> out = oiio::ImageOutput::create(path);
  if (!out) {
    return false;
  }

  oiio::ImageSpec spec(_memoryWidth, _memoryHeight, 4, oiio::TypeDesc::FLOAT);
  spec.tile_width = _tileSize;
  spec.tile_height = _tileSize;
  
  if (!out->open(path, spec)) {
    return false;
  }

  for (int i = 0; i < _tilesArray.size(); i++) {

    std::vector<image::CachedTile::smart_pointer> & row = _tilesArray[i];

    for (int j = 0; j < row.size(); j++) {

      if (!row[j]->acquire()) {
        return false;
      }

      unsigned char * ptr = row[j]->getDataPointer();

      out->write_tile(j * _tileSize, i * _tileSize, 0, oiio::TypeDesc::FLOAT, ptr);
    }
  }

  out->close();

  return true;
}

class Compositer {
public:
  Compositer(std::shared_ptr<image::TileCacheManager> manager, size_t outputWidth, size_t outputHeight) :
  _tileManager(manager),
  _panoramaWidth(outputWidth), 
  _panoramaHeight(outputHeight)
  {
  }

  virtual bool initialize() {
    
    if (!_panorama.createImage(_tileManager, _panoramaWidth, _panoramaHeight)) {
      return false;
    }

    if (!_panorama.fill(image::RGBAfColor(0.0f, 0.0f, 0.0f, 0.0f))) {
      return false;
    }
    
    return true;
  }

  virtual bool append(const aliceVision::image::Image<image::RGBfColor> & color, const aliceVision::image::Image<unsigned char> & inputMask, const aliceVision::image::Image<float> & inputWeights, size_t offset_x, size_t offset_y) {
    
    std::vector<std::vector<image::CachedTile::smart_pointer>> & img = _panorama.getTiles();

    for (int i = 0; i < img.size(); i++) {

      int top = i * _tileManager->getTileHeight() - offset_y;
      int bottom = (i + 1) * _tileManager->getTileHeight() - offset_y;
      if (top >= color.Height()) continue;
      if (bottom < 0) continue;

      std::vector<image::CachedTile::smart_pointer> & row = img[i];

      for (int j = 0; j < row.size(); j++) {

        int left = j * _tileManager->getTileWidth() - offset_x;
        int right = (j + 1) * _tileManager->getTileWidth() - offset_x;

        if (left >= color.Width()) continue;
        if (right < 0) {
          if (left + _panoramaWidth >= color.Width()) {
            continue;
          }
        }

        image::CachedTile::smart_pointer ptr = row[j];
        if (!ptr) continue;

        
        if (!ptr->acquire()) {
          continue;
        }

        image::RGBAfColor * outputptr = (image::RGBAfColor *)ptr->getDataPointer();

        #pragma omp parallel
        for (int y = 0; y < _tileManager->getTileHeight(); y++) {
          
          int ry = top + y;
          if (ry < 0 || ry >= color.Height()) {

            continue;
          }
          
          image::RGBAfColor * rowptr = outputptr + _tileManager->getTileWidth() * y;

          for (int x = 0; x < _tileManager->getTileWidth(); x++) {

            int rx = left + x;
            if (rx < 0 || rx >= color.Width()) {

              rx = _panoramaWidth + rx;
              if (rx < 0 || rx >= color.Width()) {
                continue;
              }
            }
          
            if (!inputMask(ry, rx)) continue;

            image::RGBfColor pix = color(ry, rx);

            rowptr[x].r() = pix.r();
            rowptr[x].g() = pix.g();
            rowptr[x].b() = pix.b();
            rowptr[x].a() = 1.0;
          }
        }
      }
    }

    return true;
  }

  virtual bool terminate() {
    return true;
  }

  bool save(const std::string & outputPath) {

    return _panorama.writeImage(outputPath);
  }

protected:
  std::shared_ptr<image::TileCacheManager> _tileManager;
  CachedImage<image::RGBAfColor> _panorama;
  size_t _panoramaWidth;
  size_t _panoramaHeight;
};

int aliceVision_main(int argc, char **argv)
{
  std::string sfmDataFilepath;
  std::string warpingFolder;
  std::string outputPanorama;

  std::string compositerType = "multiband";
  std::string overlayType = "none";
  bool useGraphCut = true;
  bool showBorders = false;
  bool showSeams = false;

  image::EStorageDataType storageDataType = image::EStorageDataType::Float;

  system::EVerboseLevel verboseLevel = system::Logger::getDefaultVerboseLevel();

  // Program description
  po::options_description allParams (
    "Perform panorama stiching of cameras around a nodal point for 360° panorama creation. \n"
    "AliceVision PanoramaCompositing"
  );

  // Description of mandatory parameters
  po::options_description requiredParams("Required parameters");
  requiredParams.add_options()
    ("input,i", po::value<std::string>(&sfmDataFilepath)->required(), "Input sfmData.")
    ("warpingFolder,w", po::value<std::string>(&warpingFolder)->required(), "Folder with warped images.")
    ("output,o", po::value<std::string>(&outputPanorama)->required(), "Path of the output panorama.");
  allParams.add(requiredParams);

  // Description of optional parameters
  po::options_description optionalParams("Optional parameters");
  optionalParams.add_options()
    ("compositerType,c", po::value<std::string>(&compositerType)->required(), "Compositer Type [replace, alpha, multiband].")
    ("overlayType,c", po::value<std::string>(&overlayType)->required(), "Overlay Type [none, borders, seams, all].")
    ("useGraphCut,c", po::value<bool>(&useGraphCut)->default_value(useGraphCut), "Do we use graphcut for ghost removal ?")
    ("storageDataType", po::value<image::EStorageDataType>(&storageDataType)->default_value(storageDataType),
      ("Storage data type: " + image::EStorageDataType_informations()).c_str());
  allParams.add(optionalParams);

  // Setup log level given command line
  po::options_description logParams("Log parameters");
  logParams.add_options()
    ("verboseLevel,v", po::value<system::EVerboseLevel>(&verboseLevel)->default_value(verboseLevel), "verbosity level (fatal, error, warning, info, debug, trace).");
  allParams.add(logParams);


  // Effectively parse command line given parse options
  po::variables_map vm;
  try
  {
    po::store(po::parse_command_line(argc, argv, allParams), vm);

    if(vm.count("help") || (argc == 1))
    {
      ALICEVISION_COUT(allParams);
      return EXIT_SUCCESS;
    }
    po::notify(vm);
  }
  catch(boost::program_options::required_option& e)
  {
    ALICEVISION_CERR("ERROR: " << e.what());
    ALICEVISION_COUT("Usage:\n\n" << allParams);
    return EXIT_FAILURE;
  }
  catch(boost::program_options::error& e)
  {
    ALICEVISION_CERR("ERROR: " << e.what());
    ALICEVISION_COUT("Usage:\n\n" << allParams);
    return EXIT_FAILURE;
  }

  ALICEVISION_COUT("Program called with the following parameters:");
  ALICEVISION_COUT(vm);

  // Set verbose level given command line
  system::Logger::get()->setLogLevel(verboseLevel);

  if (overlayType == "borders" || overlayType == "all")
  {
    showBorders = true;
  }

  if (overlayType == "seams" || overlayType == "all") {
    showSeams = true;
  }

  // load input scene
  sfmData::SfMData sfmData;
  if(!sfmDataIO::Load(sfmData, sfmDataFilepath, sfmDataIO::ESfMData(sfmDataIO::VIEWS|sfmDataIO::EXTRINSICS|sfmDataIO::INTRINSICS)))
  {
    ALICEVISION_LOG_ERROR("The input file '" + sfmDataFilepath + "' cannot be read");
    return EXIT_FAILURE;
  }

  std::pair<int, int> panoramaSize;
  {
      const IndexT viewId = *sfmData.getValidViews().begin();
      const std::string viewFilepath = (fs::path(warpingFolder) / (std::to_string(viewId) + ".exr")).string();
      ALICEVISION_LOG_TRACE("Read panorama size from file: " << viewFilepath);

      oiio::ParamValueList metadata = image::readImageMetadata(viewFilepath);
      panoramaSize.first = metadata.find("AliceVision:panoramaWidth")->get_int();
      panoramaSize.second = metadata.find("AliceVision:panoramaHeight")->get_int();

      if(panoramaSize.first == 0 || panoramaSize.second == 0)
      {
          ALICEVISION_LOG_ERROR("The output panorama size is empty.");
          return EXIT_FAILURE;
      }
      ALICEVISION_LOG_INFO("Output panorama size set to " << panoramaSize.first << "x" << panoramaSize.second);
  }

  std::shared_ptr<image::TileCacheManager> cacheManager = image::TileCacheManager::create(boost::filesystem::path(outputPanorama).parent_path().string(), 256, 256, 65536);
  if (!cacheManager) {
    ALICEVISION_LOG_ERROR("Error creating the cache manager");
    return EXIT_FAILURE;
  }

  cacheManager->setInCoreMaxObjectCount(100);
  
  std::unique_ptr<Compositer> compositer;
  bool isMultiBand = false;
  compositer = std::unique_ptr<Compositer>(new Compositer(cacheManager, panoramaSize.first, panoramaSize.second));
  if (!compositer->initialize()) {
    ALICEVISION_LOG_ERROR("Error initalizing the compositer");
    return EXIT_FAILURE;
  }

  // Compute seams
  std::vector<std::shared_ptr<sfmData::View>> viewsToDraw;
  for (auto& viewIt : sfmData.getViews())
  {
    if(!sfmData.isPoseAndIntrinsicDefined(viewIt.second.get()))
    {
        // skip unreconstructed views
        continue;
    }
    
    viewsToDraw.push_back(viewIt.second);
  }

  std::vector<std::shared_ptr<sfmData::View>> copy = viewsToDraw;


  viewsToDraw.insert(viewsToDraw.end(), copy.begin(), copy.end());  
  viewsToDraw.insert(viewsToDraw.end(), copy.begin(), copy.end());  

  /* Retrieve seams from distance tool */
  oiio::ParamValueList outputMetadata;

  //viewsToDraw.erase(viewsToDraw.begin(), viewsToDraw.begin() + viewsToDraw.size() - 5);

  // Do compositing
  for (const auto & view : viewsToDraw)
  {
    IndexT viewId = view->getViewId();

    if(!sfmData.isPoseAndIntrinsicDefined(view.get()))
    {
        // skip unreconstructed views
        continue;
    }

    // Load image and convert it to linear colorspace
    const std::string imagePath = (fs::path(warpingFolder) / (std::to_string(viewId) + ".exr")).string();
    ALICEVISION_LOG_INFO("Load image with path " << imagePath);
    image::Image<image::RGBfColor> source;
    image::readImage(imagePath, source, image::EImageColorSpace::NO_CONVERSION);

    oiio::ParamValueList metadata = image::readImageMetadata(imagePath);
    if(outputMetadata.empty())
    {
        // the first one will define the output metadata (random selection)
        outputMetadata = metadata;
    }
    const std::size_t offsetX = metadata.find("AliceVision:offsetX")->get_int();
    const std::size_t offsetY = metadata.find("AliceVision:offsetY")->get_int();

    // Load mask
    const std::string maskPath = (fs::path(warpingFolder) / (std::to_string(viewId) + "_mask.exr")).string();
    ALICEVISION_LOG_INFO("Load mask with path " << maskPath);
    image::Image<unsigned char> mask;
    image::readImage(maskPath, mask, image::EImageColorSpace::NO_CONVERSION);

    // Load Weights
    const std::string weightsPath = (fs::path(warpingFolder) / (std::to_string(viewId) + "_weight.exr")).string();
    ALICEVISION_LOG_INFO("Load weights with path " << weightsPath);
    image::Image<float> weights;
    image::readImage(weightsPath, weights, image::EImageColorSpace::NO_CONVERSION);


    compositer->append(source, mask, weights, offsetX, offsetY);
  }

  // Build image
  compositer->terminate();


  // Remove Warping-specific metadata
  outputMetadata.remove("AliceVision:offsetX");
  outputMetadata.remove("AliceVision:offsetY");
  outputMetadata.remove("AliceVision:panoramaWidth");
  outputMetadata.remove("AliceVision:panoramaHeight");
  // no notion of extra orientation on the output panorama
  outputMetadata.remove("Orientation");
  outputMetadata.remove("orientation");

  // Store output
  //ALICEVISION_LOG_INFO("Write output panorama to file " << outputPanorama);
  //const aliceVision::image::Image<image::RGBAfColor> & panorama = compositer->getPanorama();

  // Select storage data type
  outputMetadata.push_back(oiio::ParamValue("AliceVision:storageDataType", image::EStorageDataType_enumToString(storageDataType)));

  compositer->save(outputPanorama);
  //image::writeImage(outputPanorama, panorama, image::EImageColorSpace::AUTO, outputMetadata);

  return EXIT_SUCCESS;
}
