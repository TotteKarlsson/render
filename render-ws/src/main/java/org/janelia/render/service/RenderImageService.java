package org.janelia.render.service;

import java.io.File;
import java.net.UnknownHostException;
import java.nio.file.Paths;

import javax.ws.rs.Consumes;
import javax.ws.rs.GET;
import javax.ws.rs.PUT;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Request;
import javax.ws.rs.core.Response;

import org.janelia.alignment.RenderParameters;
import org.janelia.alignment.Utils;
import org.janelia.alignment.spec.Bounds;
import org.janelia.alignment.spec.stack.StackId;
import org.janelia.alignment.spec.stack.StackMetaData;
import org.janelia.alignment.spec.stack.StackStats;
import org.janelia.render.service.util.RenderServiceUtil;
import org.janelia.render.service.util.ResponseHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;

/**
 * APIs for rendering images server-side.
 *
 * @author Eric Trautman
 */
@Path("/v1/owner/{owner}")
@Api(tags = {"Image APIs"})
public class RenderImageService {

    private final RenderDataService renderDataService;

    @SuppressWarnings("UnusedDeclaration")
    public RenderImageService()
            throws UnknownHostException {
        this(new RenderDataService());
    }

    public RenderImageService(final RenderDataService renderDataService) {
        this.renderDataService = renderDataService;
    }

    @Path("jpeg-image")
    @PUT
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(RenderServiceUtil.IMAGE_JPEG_MIME_TYPE)
    @ApiOperation(
            tags = "Spec Image APIs",
            value = "Render JPEG image from a provide spec")
    public Response renderJpegImageFromProvidedParameters(final RenderParameters renderParameters) {
        return RenderServiceUtil.renderImageStream(renderParameters,
                                                   Utils.JPEG_FORMAT,
                                                   RenderServiceUtil.IMAGE_JPEG_MIME_TYPE,
                                                   null,
                                                   ResponseHelper.NO_CACHE_HELPER);
    }


    @Path("png-image")
    @PUT
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(RenderServiceUtil.IMAGE_PNG_MIME_TYPE)
    @ApiOperation(
            tags = "Spec Image APIs",
            value = "Render PNG image from a provide spec")
    public Response renderPngImageFromProvidedParameters(final RenderParameters renderParameters) {
        return RenderServiceUtil.renderImageStream(renderParameters,
                                                   Utils.PNG_FORMAT,
                                                   RenderServiceUtil.IMAGE_PNG_MIME_TYPE,
                                                   null,
                                                   ResponseHelper.NO_CACHE_HELPER);
    }

    @Path("tiff-image")
    @PUT
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(RenderServiceUtil.IMAGE_TIFF_MIME_TYPE)
    @ApiOperation(
            tags = "Spec Image APIs",
            value = "Render TIFF image from a provide spec")
    public Response renderTiffImageFromProvidedParameters(final RenderParameters renderParameters) {
        return RenderServiceUtil.renderImageStream(renderParameters,
                                                   Utils.TIFF_FORMAT,
                                                   RenderServiceUtil.IMAGE_TIFF_MIME_TYPE,
                                                   null,
                                                   ResponseHelper.NO_CACHE_HELPER);
    }

    @Path("project/{project}/stack/{stack}/z/{z}/jpeg-image")
    @GET
    @Produces(RenderServiceUtil.IMAGE_JPEG_MIME_TYPE)
    @ApiOperation(
            tags = "Section Image APIs",
            value = "Render JPEG image for a section")
    public Response renderJpegImageForZ(@PathParam("owner") final String owner,
                                        @PathParam("project") final String project,
                                        @PathParam("stack") final String stack,
                                        @PathParam("z") final Double z,
                                        @QueryParam("scale") Double scale,
                                        @QueryParam("filter") final Boolean filter,
                                        @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                        @Context final Request request) {

        LOG.info("renderJpegImageForZ: entry, owner={}, project={}, stack={}, z={}, scale={}, filter={}",
                 owner, project, stack, z, scale, filter);

        if (scale == null) {
            scale = 0.01;
        }

        final ResponseHelper responseHelper = new ResponseHelper(request, getStackMetaData(owner, project, stack));
        if (responseHelper.isModified()) {
            final RenderParameters renderParameters =
                    renderDataService.getRenderParametersForZ(owner, project, stack, z, scale, filter);
            return RenderServiceUtil.renderJpegImage(renderParameters, maxTileSpecsToRender, responseHelper);
        } else {
            return responseHelper.getNotModifiedResponse();
        }
    }

    @Path("project/{project}/stack/{stack}/z/{z}/png-image")
    @GET
    @Produces(RenderServiceUtil.IMAGE_PNG_MIME_TYPE)
    @ApiOperation(
            tags = "Section Image APIs",
            value = "Render PNG image for a section")
    public Response renderPngImageForZ(@PathParam("owner") final String owner,
                                       @PathParam("project") final String project,
                                       @PathParam("stack") final String stack,
                                       @PathParam("z") final Double z,
                                       @QueryParam("scale") Double scale,
                                       @QueryParam("filter") final Boolean filter,
                                       @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                       @Context final Request request) {

        LOG.info("renderPngImageForZ: entry, owner={}, project={}, stack={}, z={}, scale={}, filter={}",
                 owner, project, stack, z, scale, filter);

        if (scale == null) {
            scale = 0.01;
        }

        final ResponseHelper responseHelper = new ResponseHelper(request, getStackMetaData(owner, project, stack));
        if (responseHelper.isModified()) {
            final RenderParameters renderParameters =
                    renderDataService.getRenderParametersForZ(owner, project, stack, z, scale, filter);
            return RenderServiceUtil.renderPngImage(renderParameters, maxTileSpecsToRender, responseHelper);
        } else {
            return responseHelper.getNotModifiedResponse();
        }
    }

    @Path("project/{project}/stack/{stack}/z/{z}/tiff-image")
    @GET
    @Produces(RenderServiceUtil.IMAGE_TIFF_MIME_TYPE)
    @ApiOperation(
            tags = "Section Image APIs",
            value = "Render TIFF image for a section")
    public Response renderTiffImageForZ(@PathParam("owner") final String owner,
                                        @PathParam("project") final String project,
                                        @PathParam("stack") final String stack,
                                        @PathParam("z") final Double z,
                                        @QueryParam("scale") Double scale,
                                        @QueryParam("filter") final Boolean filter,
                                        @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                        @Context final Request request) {

        LOG.info("renderTiffImageForZ: entry, owner={}, project={}, stack={}, z={}, scale={}, filter={}",
                 owner, project, stack, z, scale, filter);

        if (scale == null) {
            scale = 0.01;
        }

        final ResponseHelper responseHelper = new ResponseHelper(request, getStackMetaData(owner, project, stack));
        if (responseHelper.isModified()) {
            final RenderParameters renderParameters =
                    renderDataService.getRenderParametersForZ(owner, project, stack, z, scale, filter);
            return RenderServiceUtil.renderTiffImage(renderParameters, maxTileSpecsToRender, responseHelper);
        } else {
            return responseHelper.getNotModifiedResponse();
        }
    }

    @Path("project/{project}/stack/{stack}/z/{z}/box/{x},{y},{width},{height},{scale}/jpeg-image")
    @GET
    @Produces(RenderServiceUtil.IMAGE_JPEG_MIME_TYPE)
    @ApiOperation(
            tags = "Bounding Box Image APIs",
            value = "Render JPEG image for the specified bounding box")
    public Response renderJpegImageForBox(@PathParam("owner") final String owner,
                                          @PathParam("project") final String project,
                                          @PathParam("stack") final String stack,
                                          @PathParam("x") final Double x,
                                          @PathParam("y") final Double y,
                                          @PathParam("z") final Double z,
                                          @PathParam("width") final Integer width,
                                          @PathParam("height") final Integer height,
                                          @PathParam("scale") final Double scale,
                                          @QueryParam("filter") final Boolean filter,
                                          @QueryParam("binaryMask") final Boolean binaryMask,
                                          @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                          @Context final Request request) {

        LOG.info("renderJpegImageForBox: entry");

        final ResponseHelper responseHelper = new ResponseHelper(request, getStackMetaData(owner, project, stack));
        if (responseHelper.isModified()) {
            final RenderParameters renderParameters =
                    getRenderParametersForGroupBox(owner, project, stack, null,
                                                   x, y, z, width, height, scale, filter, binaryMask);
            return RenderServiceUtil.renderJpegImage(renderParameters, maxTileSpecsToRender, responseHelper);
        } else {
            return responseHelper.getNotModifiedResponse();
        }
    }

    @Path("project/{project}/stack/{stack}/z/{z}/box/{x},{y},{width},{height},{scale}/png-image")
    @GET
    @Produces(RenderServiceUtil.IMAGE_PNG_MIME_TYPE)
    @ApiOperation(
            tags = "Bounding Box Image APIs",
            value = "Render PNG image for the specified bounding box")
    public Response renderPngImageForBox(@PathParam("owner") final String owner,
                                         @PathParam("project") final String project,
                                         @PathParam("stack") final String stack,
                                         @PathParam("x") final Double x,
                                         @PathParam("y") final Double y,
                                         @PathParam("z") final Double z,
                                         @PathParam("width") final Integer width,
                                         @PathParam("height") final Integer height,
                                         @PathParam("scale") final Double scale,
                                         @QueryParam("filter") final Boolean filter,
                                         @QueryParam("binaryMask") final Boolean binaryMask,
                                         @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                         @Context final Request request) {

        LOG.info("renderPngImageForBox: entry");

        final ResponseHelper responseHelper = new ResponseHelper(request, getStackMetaData(owner, project, stack));
        if (responseHelper.isModified()) {
            final RenderParameters renderParameters =
                    getRenderParametersForGroupBox(owner, project, stack, null,
                                                   x, y, z, width, height, scale, filter, binaryMask);
            return RenderServiceUtil.renderPngImage(renderParameters, maxTileSpecsToRender, responseHelper);
        } else {
            return responseHelper.getNotModifiedResponse();
        }
    }

    @Path("project/{project}/stack/{stack}/z/{z}/box/{x},{y},{width},{height},{scale}/tiff-image")
    @GET
    @Produces(RenderServiceUtil.IMAGE_TIFF_MIME_TYPE)
    @ApiOperation(
            tags = "Bounding Box Image APIs",
            value = "Render TIFF image for the specified bounding box")
    public Response renderTiffImageForBox(@PathParam("owner") final String owner,
                                          @PathParam("project") final String project,
                                          @PathParam("stack") final String stack,
                                          @PathParam("x") final Double x,
                                          @PathParam("y") final Double y,
                                          @PathParam("z") final Double z,
                                          @PathParam("width") final Integer width,
                                          @PathParam("height") final Integer height,
                                          @PathParam("scale") final Double scale,
                                          @QueryParam("filter") final Boolean filter,
                                          @QueryParam("binaryMask") final Boolean binaryMask,
                                          @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                          @Context final Request request) {

        LOG.info("renderTiffImageForBox: entry");

        final ResponseHelper responseHelper = new ResponseHelper(request, getStackMetaData(owner, project, stack));
        if (responseHelper.isModified()) {
            final RenderParameters renderParameters =
                    getRenderParametersForGroupBox(owner, project, stack, null,
                                                   x, y, z, width, height, scale, filter, binaryMask);
            return RenderServiceUtil.renderTiffImage(renderParameters, maxTileSpecsToRender, responseHelper);
        } else {
            return responseHelper.getNotModifiedResponse();
        }
    }

    @Path("project/{project}/stack/{stack}/group/{groupId}/z/{z}/box/{x},{y},{width},{height},{scale}/jpeg-image")
    @GET
    @Produces(RenderServiceUtil.IMAGE_JPEG_MIME_TYPE)
    @ApiOperation(
            tags = "Bounding Box Image APIs",
            value = "Render JPEG image for the specified bounding box and groupId")
    public Response renderJpegImageForGroupBox(@PathParam("owner") final String owner,
                                               @PathParam("project") final String project,
                                               @PathParam("stack") final String stack,
                                               @PathParam("groupId") final String groupId,
                                               @PathParam("x") final Double x,
                                               @PathParam("y") final Double y,
                                               @PathParam("z") final Double z,
                                               @PathParam("width") final Integer width,
                                               @PathParam("height") final Integer height,
                                               @PathParam("scale") final Double scale,
                                               @QueryParam("filter") final Boolean filter,
                                               @QueryParam("binaryMask") final Boolean binaryMask,
                                               @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                               @Context final Request request) {

        LOG.info("renderJpegImageForGroupBox: entry");

        final ResponseHelper responseHelper = new ResponseHelper(request, getStackMetaData(owner, project, stack));
        if (responseHelper.isModified()) {
            final RenderParameters renderParameters =
                    getRenderParametersForGroupBox(owner, project, stack, groupId,
                                                   x, y, z, width, height, scale, filter, binaryMask);
            return RenderServiceUtil.renderJpegImage(renderParameters, maxTileSpecsToRender, responseHelper);
        } else {
            return responseHelper.getNotModifiedResponse();
        }
    }

    @Path("project/{project}/stack/{stack}/group/{groupId}/z/{z}/box/{x},{y},{width},{height},{scale}/png-image")
    @GET
    @Produces(RenderServiceUtil.IMAGE_PNG_MIME_TYPE)
    @ApiOperation(
            tags = "Bounding Box Image APIs",
            value = "Render PNG image for the specified bounding box and groupId")
    public Response renderPngImageForGroupBox(@PathParam("owner") final String owner,
                                              @PathParam("project") final String project,
                                              @PathParam("stack") final String stack,
                                              @PathParam("groupId") final String groupId,
                                              @PathParam("x") final Double x,
                                              @PathParam("y") final Double y,
                                              @PathParam("z") final Double z,
                                              @PathParam("width") final Integer width,
                                              @PathParam("height") final Integer height,
                                              @PathParam("scale") final Double scale,
                                              @QueryParam("filter") final Boolean filter,
                                              @QueryParam("binaryMask") final Boolean binaryMask,
                                              @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                              @Context final Request request) {

        LOG.info("renderPngImageForGroupBox: entry");

        final ResponseHelper responseHelper = new ResponseHelper(request, getStackMetaData(owner, project, stack));
        if (responseHelper.isModified()) {
            final RenderParameters renderParameters =
                    getRenderParametersForGroupBox(owner, project, stack, groupId,
                                                   x, y, z, width, height, scale, filter, binaryMask);
            return RenderServiceUtil.renderPngImage(renderParameters, maxTileSpecsToRender, responseHelper);
        } else {
            return responseHelper.getNotModifiedResponse();
        }
    }

    @Path("project/{project}/stack/{stack}/group/{groupId}/z/{z}/box/{x},{y},{width},{height},{scale}/tiff-image")
    @GET
    @Produces(RenderServiceUtil.IMAGE_TIFF_MIME_TYPE)
    @ApiOperation(
            tags = "Bounding Box Image APIs",
            value = "Render TIFF image for the specified bounding box and groupId")
    public Response renderTiffImageForGroupBox(@PathParam("owner") final String owner,
                                               @PathParam("project") final String project,
                                               @PathParam("stack") final String stack,
                                               @PathParam("groupId") final String groupId,
                                               @PathParam("x") final Double x,
                                               @PathParam("y") final Double y,
                                               @PathParam("z") final Double z,
                                               @PathParam("width") final Integer width,
                                               @PathParam("height") final Integer height,
                                               @PathParam("scale") final Double scale,
                                               @QueryParam("filter") final Boolean filter,
                                               @QueryParam("binaryMask") final Boolean binaryMask,
                                               @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                               @Context final Request request) {

        LOG.info("renderTiffImageForGroupBox: entry");

        final ResponseHelper responseHelper = new ResponseHelper(request, getStackMetaData(owner, project, stack));
        if (responseHelper.isModified()) {
            final RenderParameters renderParameters =
                    getRenderParametersForGroupBox(owner, project, stack, groupId,
                                                   x, y, z, width, height, scale, filter, binaryMask);
            return RenderServiceUtil.renderTiffImage(renderParameters, maxTileSpecsToRender, responseHelper);
        } else {
            return responseHelper.getNotModifiedResponse();
        }
    }

    @Path("project/{project}/stack/{stack}/largeDataTileSource/{width}/{height}/{level}/{z}/{row}/{column}.jpg")
    @GET
    @Produces(RenderServiceUtil.IMAGE_JPEG_MIME_TYPE)
    @ApiOperation(
            tags = "Bounding Box Image APIs",
            value = "Render JPEG image for the specified large data (type 5) tile")
    public Response renderLargeDataTileSourceJpeg(@PathParam("owner") final String owner,
                                                  @PathParam("project") final String project,
                                                  @PathParam("stack") final String stack,
                                                  @PathParam("width") final Integer width,
                                                  @PathParam("height") final Integer height,
                                                  @PathParam("level") final Integer level,
                                                  @PathParam("z") final Double z,
                                                  @PathParam("row") final Integer row,
                                                  @PathParam("column") final Integer column,
                                                  @QueryParam("filter") final Boolean filter,
                                                  @QueryParam("binaryMask") final Boolean binaryMask,
                                                  @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                                  @Context final Request request) {

        return renderLargeDataTileSource(owner, project, stack, width, height, level, z, row, column,
                                         Utils.JPEG_FORMAT, RenderServiceUtil.IMAGE_JPEG_MIME_TYPE,
                                         filter, binaryMask, maxTileSpecsToRender,
                                         request);
    }


    @Path("project/{project}/stack/{stack}/largeDataTileSource/{width}/{height}/small/{z}.jpg")
    @GET
    @Produces(RenderServiceUtil.IMAGE_JPEG_MIME_TYPE)
    @ApiOperation(
            tags = "Bounding Box Image APIs",
            value = "Render JPEG image for the specified large data (type 5) section overview")
    public Response renderLargeDataOverviewJpeg(@PathParam("owner") final String owner,
                                                @PathParam("project") final String project,
                                                @PathParam("stack") final String stack,
                                                @PathParam("width") final Integer width,
                                                @PathParam("height") final Integer height,
                                                @PathParam("z") final Double z,
                                                @QueryParam("overviewWidth") final Integer overviewWidth,
                                                @QueryParam("filter") final Boolean filter,
                                                @QueryParam("binaryMask") final Boolean binaryMask,
                                                @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                                @Context final Request request) {

        return renderLargeDataOverview(owner, project, stack, width, height, z,
                                       Utils.JPEG_FORMAT, RenderServiceUtil.IMAGE_JPEG_MIME_TYPE,
                                       overviewWidth, filter, binaryMask, maxTileSpecsToRender,
                                       request);
    }

    @Path("project/{project}/stack/{stack}/largeDataTileSource/{width}/{height}/{level}/{z}/{row}/{column}.png")
    @GET
    @Produces(RenderServiceUtil.IMAGE_PNG_MIME_TYPE)
    @ApiOperation(
            tags = "Bounding Box Image APIs",
            value = "Render PNG image for the specified large data (type 5) tile")
    public Response renderLargeDataTileSourcePng(@PathParam("owner") final String owner,
                                                 @PathParam("project") final String project,
                                                 @PathParam("stack") final String stack,
                                                 @PathParam("width") final Integer width,
                                                 @PathParam("height") final Integer height,
                                                 @PathParam("level") final Integer level,
                                                 @PathParam("z") final Double z,
                                                 @PathParam("row") final Integer row,
                                                 @PathParam("column") final Integer column,
                                                 @QueryParam("filter") final Boolean filter,
                                                 @QueryParam("binaryMask") final Boolean binaryMask,
                                                 @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                                 @Context final Request request) {

        return renderLargeDataTileSource(owner, project, stack, width, height, level, z, row, column,
                                         Utils.PNG_FORMAT, RenderServiceUtil.IMAGE_PNG_MIME_TYPE,
                                         filter, binaryMask, maxTileSpecsToRender,
                                         request);
    }

    @Path("project/{project}/stack/{stack}/largeDataTileSource/{width}/{height}/small/{z}.png")
    @GET
    @Produces(RenderServiceUtil.IMAGE_PNG_MIME_TYPE)
    @ApiOperation(
            tags = "Bounding Box Image APIs",
            value = "Render PNG image for the specified large data (type 5) section overview")
    public Response renderLargeDataOverviewPng(@PathParam("owner") final String owner,
                                               @PathParam("project") final String project,
                                               @PathParam("stack") final String stack,
                                               @PathParam("width") final Integer width,
                                               @PathParam("height") final Integer height,
                                               @PathParam("z") final Double z,
                                               @QueryParam("overviewWidth") final Integer overviewWidth,
                                               @QueryParam("filter") final Boolean filter,
                                               @QueryParam("binaryMask") final Boolean binaryMask,
                                               @QueryParam("maxTileSpecsToRender") final Integer maxTileSpecsToRender,
                                               @Context final Request request) {

        return renderLargeDataOverview(owner, project, stack, width, height, z,
                                       Utils.PNG_FORMAT, RenderServiceUtil.IMAGE_PNG_MIME_TYPE,
                                       overviewWidth, filter, binaryMask, maxTileSpecsToRender,
                                       request);
    }

    private Response renderLargeDataTileSource(final String owner,
                                               final String project,
                                               final String stack,
                                               final Integer width,
                                               final Integer height,
                                               final Integer level,
                                               final Double z,
                                               final Integer row,
                                               final Integer column,
                                               final String format,
                                               final String mimeType,
                                               final Boolean filter,
                                               final Boolean binaryMask,
                                               Integer maxTileSpecsToRender,
                                               final Request request) {

        LOG.info("renderLargeDataTileSource: entry, stack={}, width={}, height={}, z={}, row={}, column={}",
                 stack, width, height, z, row, column);

        final StackMetaData stackMetaData = getStackMetaData(owner, project, stack);
        final ResponseHelper responseHelper = new ResponseHelper(request, stackMetaData);
        if (responseHelper.isModified()) {

            final File sourceFile = getLargeDataFile(stackMetaData, filter, binaryMask,
                                                     width + "x" + height,
                                                     level.toString(),
                                                     String.valueOf(z.intValue()),
                                                     row.toString(),
                                                     column + "." + format);

            if (sourceFile == null) {

                final double factor = 1 << level;
                final double scaledWidth = width * factor;
                final double scaledHeight = height * factor;
                final double x = column * scaledWidth;
                final double y = row * scaledHeight;
                final double scale = 1.0 / factor;

                final RenderParameters renderParameters =
                        getRenderParametersForGroupBox(owner, project, stack, null,
                                                       x, y, z, (int) scaledWidth, (int) scaledHeight, scale,
                                                       filter, binaryMask);

                if (maxTileSpecsToRender == null) {
                    maxTileSpecsToRender = DEFAULT_MAX_TILE_SPECS_FOR_LARGE_DATA;
                }

                return RenderServiceUtil.renderImageStream(renderParameters,
                                                           format,
                                                           mimeType,
                                                           maxTileSpecsToRender,
                                                           responseHelper);

            }  else {

                return RenderServiceUtil.streamImageFile(sourceFile, mimeType, responseHelper);

            }

        } else {

            return responseHelper.getNotModifiedResponse();

        }
    }

    private Response renderLargeDataOverview(final String owner,
                                             final String project,
                                             final String stack,
                                             final Integer width,
                                             final Integer height,
                                             final Double z,
                                             final String format,
                                             final String mimeType,
                                             Integer overviewWidth,
                                             final Boolean filter,
                                             final Boolean binaryMask,
                                             Integer maxTileSpecsToRender,
                                             final Request request) {

        LOG.info("renderLargeDataOverview: entry, stack={}, width={}, height={}, z={}",
                 stack, width, height, z);

        final StackMetaData stackMetaData = getStackMetaData(owner, project, stack);
        final ResponseHelper responseHelper = new ResponseHelper(request, stackMetaData);
        if (responseHelper.isModified()) {

            final File overviewSourceFile = getLargeDataFile(stackMetaData, filter, binaryMask,
                                                             width + "x" + height,
                                                             "small",
                                                             String.valueOf(z.intValue()) + "." + format);

            if (overviewSourceFile == null) {

                int stackWidth = 1;
                int stackHeight = 1;

                final StackStats stats = stackMetaData.getStats();
                if (stats != null) {
                    final Bounds stackBounds = stats.getStackBounds();
                    if (stackBounds != null) {
                        stackWidth = stackBounds.getMaxX().intValue();
                        stackHeight = stackBounds.getMaxY().intValue();
                    }
                }

                final double x = 0;
                final double y = 0;
                if ((overviewWidth == null) || (overviewWidth < 1)) {
                    overviewWidth = 192;
                }
                final double scale = (double) overviewWidth / stackWidth;

                final RenderParameters renderParameters =
                        getRenderParametersForGroupBox(owner, project, stack, null,
                                                       x, y, z, stackWidth, stackHeight, scale,
                                                       filter, binaryMask);

                if (maxTileSpecsToRender == null) {
                    maxTileSpecsToRender = DEFAULT_MAX_TILE_SPECS_FOR_LARGE_DATA;
                }

                return RenderServiceUtil.renderImageStream(renderParameters,
                                                           format,
                                                           mimeType,
                                                           maxTileSpecsToRender,
                                                           responseHelper);

            }  else {

                return RenderServiceUtil.streamImageFile(overviewSourceFile, mimeType, responseHelper);

            }

        } else {

            return responseHelper.getNotModifiedResponse();

        }
    }

    private File getLargeDataFile(final StackMetaData stackMetaData,
                                  final Boolean filter,
                                  final Boolean binaryMask,
                                  final String... additionalPathElements) {

        File file = null;

        final String rootPath = stackMetaData.getCurrentMaterializedBoxRootPath();
        if (rootPath != null) {

            file = Paths.get(rootPath, additionalPathElements).toFile();

            queueLargeDataFileRequest(file, filter, binaryMask);

            if (! file.exists()) {
                // force dynamic rendering if materialized box does not exist
                file = null;
            }

        }

        return file;
    }

    private void queueLargeDataFileRequest(final File file,
                                           final Boolean filter,
                                           final Boolean binaryMask) {
        // TODO: queue (POST) request for materialize service (will either generate file or just mark usage)
        LOG.info("need to queue materialize service request for {}, filter={}, binaryMask={}",
                 file, filter, binaryMask);
    }

    private RenderParameters getRenderParametersForGroupBox(final String owner,
                                                            final String project,
                                                            final String stack,
                                                            final String groupId,
                                                            final Double x,
                                                            final Double y,
                                                            final Double z,
                                                            final Integer width,
                                                            final Integer height,
                                                            final Double scale,
                                                            final Boolean filter,
                                                            final Boolean binaryMask) {

        final StackId stackId = new StackId(owner, project, stack);
        final RenderParameters renderParameters = renderDataService.getInternalRenderParameters(stackId,
                                                                                                groupId,
                                                                                                x,
                                                                                                y,
                                                                                                z,
                                                                                                width,
                                                                                                height,
                                                                                                scale);
        renderParameters.setDoFilter(filter);
        renderParameters.setBinaryMask(binaryMask);

        return renderParameters;
    }

    private StackMetaData getStackMetaData(final String owner,
                                           final String project,
                                           final String stack) {
        final StackId stackId = new StackId(owner, project, stack);
        return renderDataService.getStackMetaData(stackId);
    }

    private static final Logger LOG = LoggerFactory.getLogger(RenderImageService.class);

    private static final Integer DEFAULT_MAX_TILE_SPECS_FOR_LARGE_DATA = 20;
}