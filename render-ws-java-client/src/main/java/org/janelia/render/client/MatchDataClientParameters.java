package org.janelia.render.client;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

/**
 * Base parameters for all match web service clients.
 *
 * @author Eric Trautman
 */
@Parameters
public class MatchDataClientParameters
        extends CommandLineParameters {

    @Parameter(
            names = "--baseDataUrl",
            description = "Base web service URL for data (e.g. http://host[:port]/render-ws/v1)",
            required = true)
    public String baseDataUrl;

    @Parameter(
            names = "--owner",
            description = "Match collection owner",
            required = true)
    public String owner;

    @Parameter(
            names = "--collection",
            description = "Match collection name",
            required = true)
    public String collection;

    public MatchDataClientParameters() {
        this.baseDataUrl = null;
        this.owner = null;
        this.collection = null;
    }

}