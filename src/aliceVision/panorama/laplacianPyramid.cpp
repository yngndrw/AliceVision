#include "laplacianPyramid.hpp"

#include "feathering.hpp"
#include "gaussian.hpp"
#include "compositer.hpp"

namespace aliceVision
{

LaplacianPyramid::LaplacianPyramid(size_t base_width, size_t base_height, size_t max_levels) :
_baseWidth(base_width),
_baseHeight(base_height),
_maxLevels(max_levels)
{
}



bool LaplacianPyramid::initialize(image::TileCacheManager::shared_ptr & cacheManager) 
{
    size_t width = _baseWidth;
    size_t height = _baseHeight;

    /*Make sure pyramid size can be divided by 2 on each levels*/
    double max_scale = 1.0 / pow(2.0, _maxLevels - 1);

    /*Prepare pyramid*/
    for(int lvl = 0; lvl < _maxLevels; lvl++)
    {
        CachedImage<image::RGBfColor> color;
        CachedImage<float> weights;

        if(!color.createImage(cacheManager, width, height))
        {
            return false;
        }

        if(!weights.createImage(cacheManager, width, height))
        {
            return false;
        }

        if(!color.fill(image::RGBfColor(0.0f, 0.0f, 0.0f)))
        {
            return false;
        }

        if(!weights.fill(0.0f))
        {
            return false;
        }

        _levels.push_back(color);
        _weights.push_back(weights);

        height /= 2;
        width /= 2;
    }

    return true;
}

bool LaplacianPyramid::augment(image::TileCacheManager::shared_ptr & cacheManager, size_t newMaxLevels)
{
    if(newMaxLevels <= _levels.size())
    {
        return false;
    }
    _maxLevels = newMaxLevels;

    //Get content of last level of pyramid
    CachedImage<image::RGBfColor> lastColor = _levels[_levels.size() - 1];
    CachedImage<float> largerWeight = _weights[_weights.size() - 1];

    //Remove last level
    /*_levels.pop_back();
    _weights.pop_back();*/

    //Last level was multiplied by the weight. 
    //Remove this factor
    lastColor.perPixelOperation(largerWeight, 
        [](const image::RGBfColor & c, const float & w) -> image::RGBfColor 
        {
            if (w < 1e-6) 
            {
                return image::RGBfColor(0.0f, 0.0f, 0.0f);
            }

            image::RGBfColor r;

            r.r() = c.r() / w;
            r.g() = c.g() / w;
            r.b() = c.b() / w;

            return r;
        }
    );

    //Create a mask
    CachedImage<unsigned char> largerMask;
    if(!largerMask.createImage(cacheManager, largerWeight.getWidth(), largerWeight.getHeight()))
    {
        return false;
    }

    //Build the mask
    largerMask.perPixelOperation(largerWeight, 
        [](const unsigned char & c, const float & w) -> unsigned char
        {
            if (w < 1e-6) 
            {
                return 0;
            }

            return 255;
        }
    );

    largerMask.writeImage("/home/mmoc/mask.exr");


    /*int largerLevel = _levels.size() - 1;
    int currentLevel = _levels.size();

    CachedImage<image::RGBfColor> largerColor = _levels[largerLevel];
    CachedImage<float> largerWeight = _weights[largerLevel];

    int width = largerColor.getWidth();
    int height = largerColor.getHeight();

    CachedImage<image::RGBfColor> color;
    if(!color.createImage(cacheManager, width, height))
    {
        return false;
    }

    CachedImage<float> weights;
    if(!weights.createImage(cacheManager, width, height))
    {
        return false;
    }

    aliceVision::image::Image<image::RGBfColor> extractedColor(width, height);
    aliceVision::image::Image<float> extractedWeight(width, height);

    BoundingBox extractBb;
    extractBb.left = 0;
    extractBb.top = 0;
    extractBb.width = width;
    extractBb.height = height;


    if (!loopyCachedImageExtract(extractedColor, largerColor, extractBb)) 
    {
        return false;
    }

    if (!loopyCachedImageExtract(extractedWeight, largerWeight, extractBb)) 
    {
        return false;
    }

    //image was multiplied with a weight, we need to get back the original weight

    _levels.push_back(color);
    _weights.push_back(weights);*/
    
    return true;
}

bool LaplacianPyramid::apply(const aliceVision::image::Image<image::RGBfColor>& source,
                             const aliceVision::image::Image<unsigned char>& mask,
                             const aliceVision::image::Image<float>& weights, size_t offset_x, size_t offset_y)
{
    int width = source.Width();
    int height = source.Height();

    /* Convert mask to alpha layer */
    image::Image<float> mask_float(width, height);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            if(mask(i, j))
            {
                mask_float(i, j) = 1.0f;
            }
            else
            {
                mask_float(i, j) = 0.0f;
            }
        }
    }

    image::Image<image::RGBfColor> current_color = source;
    image::Image<image::RGBfColor> next_color;
    image::Image<float> current_weights = weights;
    image::Image<float> next_weights;
    image::Image<float> current_mask = mask_float;
    image::Image<float> next_mask;


    for(int l = 0; l < _levels.size() - 1; l++)
    {
        aliceVision::image::Image<image::RGBfColor> buf_masked(width, height);
        aliceVision::image::Image<image::RGBfColor> buf(width, height);
        aliceVision::image::Image<image::RGBfColor> buf2(width, height);
        aliceVision::image::Image<float> buf_float(width, height);

        next_color = aliceVision::image::Image<image::RGBfColor>(width / 2, height / 2);
        next_weights = aliceVision::image::Image<float>(width / 2, height / 2);
        next_mask = aliceVision::image::Image<float>(width / 2, height / 2);

        /*Apply mask to content before convolution*/
        for(int i = 0; i < current_color.Height(); i++)
        {
            for(int j = 0; j < current_color.Width(); j++)
            {
                if(std::abs(current_mask(i, j)) > 1e-6)
                {
                    buf_masked(i, j) = current_color(i, j);
                }
                else
                {
                    buf_masked(i, j).r() = 0.0f;
                    buf_masked(i, j).g() = 0.0f;
                    buf_masked(i, j).b() = 0.0f;
                    current_weights(i, j) = 0.0f;
                }
            }
        }

        convolveGaussian5x5<image::RGBfColor>(buf, buf_masked);
        convolveGaussian5x5<float>(buf_float, current_mask);

        /*
        Normalize given mask
        */
        for(int i = 0; i < current_color.Height(); i++)
        {
            for(int j = 0; j < current_color.Width(); j++)
            {

                float m = buf_float(i, j);

                if(std::abs(m) > 1e-6)
                {
                    buf(i, j).r() = buf(i, j).r() / m;
                    buf(i, j).g() = buf(i, j).g() / m;
                    buf(i, j).b() = buf(i, j).b() / m;
                    buf_float(i, j) = 1.0f;
                }
                else
                {
                    buf(i, j).r() = 0.0f;
                    buf(i, j).g() = 0.0f;
                    buf(i, j).b() = 0.0f;
                    buf_float(i, j) = 0.0f;
                }
            }
        }

        downscale(next_color, buf);
        downscale(next_mask, buf_float);

        upscale(buf, next_color);
        convolveGaussian5x5<image::RGBfColor>(buf2, buf);

        for(int i = 0; i < buf2.Height(); i++)
        {
            for(int j = 0; j < buf2.Width(); j++)
            {
                buf2(i, j) *= 4.0f;
            }
        }

        substract(current_color, current_color, buf2);

        convolveGaussian5x5<float>(buf_float, current_weights);
        downscale(next_weights, buf_float);

        merge(current_color, current_weights, l, offset_x, offset_y);

        current_color = next_color;
        current_weights = next_weights;
        current_mask = next_mask;

        width /= 2;
        height /= 2;
        offset_x /= 2;
        offset_y /= 2;
    }

    merge(current_color, current_weights, _levels.size() - 1, offset_x, offset_y);

    return true;
}


bool LaplacianPyramid::merge(const aliceVision::image::Image<image::RGBfColor>& oimg,
                             const aliceVision::image::Image<float>& oweight, size_t level, size_t offset_x,
                             size_t offset_y)
{
    CachedImage<image::RGBfColor> & img = _levels[level];
    CachedImage<float> & weight = _weights[level];

    aliceVision::image::Image<image::RGBfColor> extractedColor(oimg.Width(), oimg.Height());
    aliceVision::image::Image<float> extractedWeight(oimg.Width(), oimg.Height());


    BoundingBox extractBb;
    extractBb.left = offset_x;
    extractBb.top = offset_y;
    extractBb.width = oimg.Width();
    extractBb.height = oimg.Height();


    if (!loopyCachedImageExtract(extractedColor, img, extractBb)) 
    {
        return false;
    }

    if (!loopyCachedImageExtract(extractedWeight, weight, extractBb)) 
    {
        return false;
    }
    
    for(int i = 0; i < oimg.Height(); i++)
    {
        for(int j = 0; j < oimg.Width(); j++)
        {
            extractedColor(i, j).r() += oimg(i, j).r() * oweight(i, j);
            extractedColor(i, j).g() += oimg(i, j).g() * oweight(i, j);
            extractedColor(i, j).b() += oimg(i, j).b() * oweight(i, j);
            extractedWeight(i, j) += oweight(i, j);
        }
    }

    BoundingBox inputBb;
    inputBb.left = 0;
    inputBb.top = 0;
    inputBb.width = extractedColor.Width();
    inputBb.height = extractedColor.Height();

    if (!loopyCachedImageAssign(img, extractedColor, extractBb, inputBb)) {
        return false;
    }

    
    if (!loopyCachedImageAssign(weight, extractedWeight, extractBb, inputBb)) {
        return false;
    }

    return true;
}

bool LaplacianPyramid::rebuild(CachedImage<image::RGBAfColor>& output)
{

    // We first want to compute the final pixels mean
    for(int l = 0; l < _levels.size(); l++)
    {
        _levels[l].perPixelOperation(_weights[l], 
            [](const image::RGBfColor & c, const float & w) -> image::RGBfColor 
            {
                if (w < 1e-6) 
                {
                    return image::RGBfColor(0.0f, 0.0f, 0.0f);
                }

                image::RGBfColor r;

                r.r() = c.r() / w;
                r.g() = c.g() / w;
                r.b() = c.b() / w;

                return r;
            }
        );
    }

    
    removeNegativeValues(_levels[_levels.size() - 1]);

    for(int l = _levels.size() - 2; l >= 0; l--)
    {
        const size_t processingSize = 512;
        const size_t borderSize = 5;

        int halfLevel = l + 1;
        int currentLevel = l;

        int x = 0;
        int y = 0;
        
        
        for (int y = 0; y < _levels[halfLevel].getHeight(); y += processingSize)
        {
            for (int x = 0; x < _levels[halfLevel].getWidth(); x += processingSize) 
            {
                BoundingBox extractedBb;
                extractedBb.left = x;
                extractedBb.top = y;
                extractedBb.width = processingSize;
                extractedBb.height = processingSize;
                extractedBb.clampLeft();
                extractedBb.clampTop();
                extractedBb.clampRight(_levels[halfLevel].getWidth() - 1);
                extractedBb.clampBottom(_levels[halfLevel].getHeight() - 1);
                
                BoundingBox dilatedBb = extractedBb.dilate(borderSize);
                dilatedBb.clampLeft();
                dilatedBb.clampTop();   
                dilatedBb.clampBottom(_levels[halfLevel].getHeight() - 1);            
                
                BoundingBox doubleDilatedBb = dilatedBb.doubleSize();
                BoundingBox doubleBb = extractedBb.doubleSize();

                aliceVision::image::Image<image::RGBfColor> extracted(dilatedBb.width, dilatedBb.height);
                if (!loopyCachedImageExtract(extracted, _levels[halfLevel], dilatedBb)) 
                {
                    return false;
                }

                aliceVision::image::Image<image::RGBfColor> extractedNext(doubleDilatedBb.width, doubleDilatedBb.height);
                if (!loopyCachedImageExtract(extractedNext, _levels[currentLevel], doubleDilatedBb)) 
                {
                    return false;
                }

                aliceVision::image::Image<image::RGBfColor> buf(doubleDilatedBb.width, doubleDilatedBb.height);
                aliceVision::image::Image<image::RGBfColor> buf2(doubleDilatedBb.width, doubleDilatedBb.height);

                upscale(buf, extracted);
                convolveGaussian5x5<image::RGBfColor>(buf2, buf, false);

                for(int i = 0; i < buf2.Height(); i++)
                {
                    for(int j = 0; j < buf2.Width(); j++)
                    {
                        buf2(i, j) *= 4.0f;
                    }
                }

                addition(extractedNext, extractedNext, buf2);

                BoundingBox inputBb;
                inputBb.left = doubleBb.left - doubleDilatedBb.left;
                inputBb.top = doubleBb.top - doubleDilatedBb.top;
                inputBb.width = doubleBb.width;
                inputBb.height = doubleBb.height;

                if (!loopyCachedImageAssign(_levels[currentLevel], extractedNext, doubleBb, inputBb)) 
                {
                    return false;
                }
            }
        }

        removeNegativeValues(_levels[currentLevel]);
    }

    for(int i = 0; i < output.getTiles().size(); i++)
    {

        std::vector<image::CachedTile::smart_pointer> & rowOutput = output.getTiles()[i];
        std::vector<image::CachedTile::smart_pointer> & rowInput = _levels[0].getTiles()[i];
        std::vector<image::CachedTile::smart_pointer> & rowWeight = _weights[0].getTiles()[i];

        for(int j = 0; j < rowOutput.size(); j++)
        {

            if(!rowOutput[j]->acquire())
            {
                return false;
            }

            if(!rowInput[j]->acquire())
            {
                return false;
            }

            if(!rowWeight[j]->acquire())
            {
                return false;
            }

            image::RGBAfColor* ptrOutput = (image::RGBAfColor *)rowOutput[j]->getDataPointer();
            image::RGBfColor* ptrInput = (image::RGBfColor *)rowInput[j]->getDataPointer();
            float* ptrWeight = (float *)rowWeight[j]->getDataPointer();

            for (int k = 0; k < output.getTileSize() * output.getTileSize(); k++) 
            {
                ptrOutput[k].r() = ptrInput[k].r();
                ptrOutput[k].g() = ptrInput[k].g();
                ptrOutput[k].b() = ptrInput[k].b();

                if(ptrWeight[k] < 1e-6)
                {
                    ptrOutput[k].a() = 1.0f;
                }
                else
                {
                    ptrOutput[k].a() = 1.0f;
                }
            }
        }
    }

    return true;
}

} // namespace aliceVision
