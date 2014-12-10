/**
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package org.janelia.alignment;

import ij.ImagePlus;
import ij.process.ImageProcessor;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.io.FileWriter;
import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.lang.Math;

import mpicbg.imagefeatures.Feature;
import mpicbg.imagefeatures.FloatArray2DSIFT;
import mpicbg.ij.SIFT;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import ij.process.ImageStatistics;
import ij.measure.Calibration;
import ij.measure.Measurements;
import ij.plugin.ContrastEnhancer;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

/**
 * 
 * @author Seymour Knowles-Barley
 */
public class ComputeSiftFeaturesFromPath
{
	@Parameters
	static private class Params
	{
	  
    
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--imageFile", description = "URL to JSON tile spec", required = true )
        private String imageFile;
        
        @Parameter( names = "--initialSigma", description = "Initial Gaussian blur sigma", required = false )
        private float initialSigma = 1.6f;
        
        @Parameter( names = "--percentSaturated", description = "Percentage of pixels to saturate when normalizing contrast before extracting features", required = false )
        private float percentSaturated = 0.5f;
        
        @Parameter( names = "--steps", description = "Steps per scale octave", required = false )
        private int steps = 3;
        
        @Parameter( names = "--minOctaveSize", description = "Min image size", required = false )
        private int minOctaveSize = 64;
        
        @Parameter( names = "--maxOctaveSize", description = "Max image size", required = false )
        private int maxOctaveSize = 0;
        
        @Parameter( names = "--fdSize", description = "Feature descriptor size", required = false )
        private int fdSize = 8;
        
        @Parameter( names = "--fdBins", description = "Feature descriptor bins", required = false )
        private int fdBins = 8;
        
        @Parameter( names = "--outputFile", description = "Path to the target image if any", required = true )
        public String outputFile = null;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
 

	}
	
	private ComputeSiftFeaturesFromPath() {}
/*	
	private static float[] translateBoundingBox( float[] originalBoundingBox, CoordinateTransformList< CoordinateTransform > ctlMipmap )
	{
		float[][] points = {
				{ originalBoundingBox[0], originalBoundingBox[2] },
				{ originalBoundingBox[0], originalBoundingBox[3] },
				{ originalBoundingBox[0], originalBoundingBox[2] },
		};
	
	}
*/
	
	public static void main( final String[] args )
	{
		
		final Params params = new Params();
		try
        {
			final JCommander jc = new JCommander( params, args );
        	if ( params.help )
            {
        		jc.usage();
                return;
            }
        }
        catch ( final Exception e )
        {
        	e.printStackTrace();
            final JCommander jc = new JCommander( params );
        	jc.setProgramName( "java [-options] -cp render.jar org.janelia.alignment.RenderTile" );
        	jc.usage(); 
        	return;
        }
		
	
						
		List< FeatureSpec > feature_data = new ArrayList< FeatureSpec >();
		
		final ImagePlus imp = Utils.openImagePlus( params.imageFile );
    final List< Feature > fs = new ArrayList< Feature >();
    if ( imp == null )
          System.err.println( "Failed to load image '" +  params.imageFile + "'." );
    else
        {
            ImageProcessor ip=imp.getProcessor();
            //enhance contrast of image
            Calibration cal = new Calibration(imp);
            ImageStatistics reference_stats = ImageStatistics.getStatistics(ip, Measurements.MIN_MAX, cal);
            ContrastEnhancer cenh=new ContrastEnhancer();
          
            cenh.setNormalize(true);
            cenh.stretchHistogram(ip,params.percentSaturated,reference_stats);
    
            /* calculate sift features for the image or sub-region */
            System.out.println( "Calculating SIFT features for image '" + params.imageFile + "'." );
            FloatArray2DSIFT.Param siftParam = new FloatArray2DSIFT.Param();
            siftParam.initialSigma = params.initialSigma;
            siftParam.steps = params.steps;
            siftParam.minOctaveSize = params.minOctaveSize;
            int maxsize=params.maxOctaveSize;
            if (params.maxOctaveSize==0){
              maxsize = (int) Math.min(imp.getHeight()/4,imp.getWidth()/4);
            }            
            siftParam.maxOctaveSize = maxsize;
            siftParam.fdSize = params.fdSize;
            siftParam.fdBins = params.fdBins;
            FloatArray2DSIFT sift = new FloatArray2DSIFT(siftParam);
            SIFT ijSIFT = new SIFT(sift);



            ijSIFT.extractFeatures( ip, fs );
            System.out.println( "found " + fs.size() + " features in " + params.imageFile );
        }
		try {
		  File file = new File(params.outputFile);
		  file.getParentFile().mkdirs();
			Writer writer = new FileWriter(file);
      //Gson gson = new GsonBuilder().create();
      Gson gson = new GsonBuilder().setPrettyPrinting().create();
      gson.toJson(fs, writer);
      writer.close();
	    }
		catch ( final IOException e )
		{
			System.err.println( "Error writing JSON file: " + params.outputFile );
			e.printStackTrace( System.err );
		}
	}
}
