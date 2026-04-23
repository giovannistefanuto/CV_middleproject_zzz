#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <imageIterator.h>
#include <featureList.h>
#include <algorithm>
#include <cmath>
#include <vector>

static void detectSIFTPoints(
    const cv::Mat& gray,
    cv::Ptr<cv::SIFT>& sift,
    std::vector<cv::Point2f>& points)
{
    // Estrae keypoint SIFT e li converte in punti 2D tracciabili con Optical Flow.
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        points.push_back(keypoints[i].pt);
    }
}

static void drawBoundingBoxFromPoints(cv::Mat& image, const std::vector<cv::Point2f>& points)
{
    if (points.empty())
    {
        return;
    }

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;

    for (size_t i = 1; i < points.size(); ++i)
    {
        minX = std::min(minX, points[i].x);
        maxX = std::max(maxX, points[i].x);
        minY = std::min(minY, points[i].y);
        maxY = std::max(maxY, points[i].y);
    }

    cv::Rect box(
        cv::Point(static_cast<int>(minX), static_cast<int>(minY)),
        cv::Point(static_cast<int>(maxX), static_cast<int>(maxY))
    );

    cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
}

static bool computeBoundingBoxFromPoints(const std::vector<cv::Point2f>& points, const cv::Size& imageSize, cv::Rect& box)
{
    // Calcola bbox axis-aligned e la limita ai bordi immagine.
    if (points.empty())
    {
        return false;
    }

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;

    for (size_t i = 1; i < points.size(); ++i)
    {
        minX = std::min(minX, points[i].x);
        maxX = std::max(maxX, points[i].x);
        minY = std::min(minY, points[i].y);
        maxY = std::max(maxY, points[i].y);
    }

    int x1 = std::max(0, static_cast<int>(minX));
    int y1 = std::max(0, static_cast<int>(minY));
    int x2 = std::min(imageSize.width - 1, static_cast<int>(maxX));
    int y2 = std::min(imageSize.height - 1, static_cast<int>(maxY));

    if (x2 <= x1 || y2 <= y1)
    {
        return false;
    }

    box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
    return true;
}

static bool processCategory(const std::string& inputFolder)
{
    // Ogni categoria salva output in una cartella dedicata accanto all'input.
    const std::string outputFolder = inputFolder + "_annotate";

    if (!cv::utils::fs::exists(inputFolder)) {
        std::cerr << "Errore: cartella input non trovata: " << inputFolder << std::endl;
        return false;
    }

    if (!cv::utils::fs::exists(outputFolder)) {
        cv::utils::fs::createDirectories(outputFolder);
    }

    const float minMovement = 0.20f;
    const float maxMovement = 100.0f;
    const int refreshEveryNFrames = 2;
    const bool showSavedFeatures = true;

    // Flag debug per categoria: abilita/disabilita i log dettagliati del movimento feature.
    const bool logBird = false;
    const bool logCar = false;
    const bool logFrog = true;
    const bool logSheep = false;
    const bool logSquirrel = false;

    bool logFeatureMotion = false;
    if (inputFolder.find("bird") != std::string::npos)
    {
        logFeatureMotion = logBird;
    }
    else if (inputFolder.find("car") != std::string::npos)
    {
        logFeatureMotion = logCar;
    }
    else if (inputFolder.find("frog") != std::string::npos)
    {
        logFeatureMotion = logFrog;
    }
    else if (inputFolder.find("sheep") != std::string::npos)
    {
        logFeatureMotion = logSheep;
    }
    else if (inputFolder.find("squirrel") != std::string::npos)
    {
        logFeatureMotion = logSquirrel;
    }

    const bool logFrameSummary = logFeatureMotion;
    const bool logCategorySummary = logFeatureMotion;

    // Iteratore immagini della categoria (supporta PNG/JPG ordinati).
    ImageIterator iterator(inputFolder);
    if (!iterator.hasNext()) {
        std::cerr << "Errore: nessuna immagine PNG/JPG trovata in " << inputFolder << std::endl;
        return false;
    }

    // Inizializzazione pipeline dal primo frame.
    cv::Mat firstFrame;
    iterator.next(firstFrame);
    if (firstFrame.empty()) {
        std::cerr << "Errore: impossibile leggere il primo frame." << std::endl;
        return false;
    }

    cv::Mat prevGray;
    cv::cvtColor(firstFrame, prevGray, cv::COLOR_BGR2GRAY);

    // Feature iniziali rilevate con SIFT sul frame 0.
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(60000);
    std::vector<cv::Point2f> activePoints;
    detectSIFTPoints(prevGray, sift, activePoints);

    cv::Rect lastBox;
    bool hasLastBox = false;
    std::vector<cv::Point2f> firstSavedPoints;

    cv::Mat pendingFrame;
    bool hasPendingFrame = false;

    // Prima box: uso Optical Flow tra frame 0 e frame 1 per massima precisione sul frame iniziale.
    if (iterator.hasNext())
    {
        iterator.next(pendingFrame);
        if (!pendingFrame.empty())
        {
            hasPendingFrame = true;

            cv::Mat pendingGray;
            cv::cvtColor(pendingFrame, pendingGray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> firstToSecondPoints;
            std::vector<uchar> firstStatus;
            std::vector<float> firstErr;

            if (!activePoints.empty())
            {
                // Stima del moto delle feature dal primo al secondo frame.
                cv::calcOpticalFlowPyrLK(prevGray, pendingGray, activePoints, firstToSecondPoints, firstStatus, firstErr);
            }

            std::vector<cv::Point2f> movingPointsInFirstFrame;
            int survivedInFirstFlow = 0;
            for (size_t i = 0; i < firstToSecondPoints.size(); ++i)
            {
                if (!firstStatus[i])
                {
                    continue;
                }

                float dx = firstToSecondPoints[i].x - activePoints[i].x;
                float dy = firstToSecondPoints[i].y - activePoints[i].y;
                float motion = std::sqrt(dx * dx + dy * dy);

                if (motion > minMovement && motion < maxMovement)
                {
                    // La box del frame 0 deve usare punti del frame 0, non del frame 1.
                    movingPointsInFirstFrame.push_back(activePoints[i]);
                    ++survivedInFirstFlow;

                    if (logFeatureMotion)
                    {
                        std::cout << "[DEBUG][Frame 0->1] feature " << i
                                  << " prev=(" << activePoints[i].x << "," << activePoints[i].y << ")"
                                  << " next=(" << firstToSecondPoints[i].x << "," << firstToSecondPoints[i].y << ")"
                                  << " motion=" << motion << std::endl;
                    }
                }
            }

            if (logFrameSummary)
            {
                std::cout << "[DEBUG][Frame 0->1] sopravvissute " << survivedInFirstFlow
                          << " su " << firstToSecondPoints.size() << std::endl;
            }

            firstSavedPoints = movingPointsInFirstFrame;

            if (computeBoundingBoxFromPoints(movingPointsInFirstFrame, firstFrame.size(), lastBox))
            {
                hasLastBox = true;
            }
        }
    }

    if (!hasLastBox)
    {
        // Fallback robusto: SIFT sul frame 0, poi full-frame solo come ultima opzione.
        if (!computeBoundingBoxFromPoints(activePoints, firstFrame.size(), lastBox))
        {
            lastBox = cv::Rect(0, 0, std::max(1, firstFrame.cols - 1), std::max(1, firstFrame.rows - 1));
        }
        if (firstSavedPoints.empty())
        {
            firstSavedPoints = activePoints;
        }
        hasLastBox = true;
    }

    // Salvataggio del primo frame annotato.
    cv::Mat firstOutput = firstFrame.clone();
    cv::rectangle(firstOutput, lastBox, cv::Scalar(0, 255, 0), 2);
    if (showSavedFeatures)
    {
        // Visualizza sul frame corrente le feature mantenute dopo i filtri.
        for (size_t i = 0; i < firstSavedPoints.size(); ++i)
        {
            cv::circle(firstOutput, firstSavedPoints[i], 2, cv::Scalar(0, 0, 255), -1);
        }
    }
    std::string firstOutputPath = outputFolder + "/frame_0000.png";
    if (!cv::imwrite(firstOutputPath, firstOutput)) {
        std::cerr << "Errore: impossibile salvare " << firstOutputPath << std::endl;
        return false;
    }

    // Loop principale: tracking frame-to-frame fino all'ultima immagine.
    int frameCounter = 1;
    while (hasPendingFrame || iterator.hasNext())
    {
        cv::Mat currentFrame;
        if (hasPendingFrame)
        {
            currentFrame = pendingFrame;
            hasPendingFrame = false;
        }
        else
        {
            iterator.next(currentFrame);
        }

        if (currentFrame.empty())
        {
            std::cerr << "Attenzione: frame vuoto al passo " << frameCounter << std::endl;
            ++frameCounter;
            continue;
        }

        cv::Mat currentGray;
        cv::cvtColor(currentFrame, currentGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> nextPoints;
        std::vector<uchar> status;
        std::vector<float> err;

        if (!activePoints.empty())
        {
            // Tracciamento LK delle feature attive tra frame consecutivi.
            cv::calcOpticalFlowPyrLK(prevGray, currentGray, activePoints, nextPoints, status, err);
        }

        // Filtro del moto: elimina sfondo quasi fermo e outlier troppo veloci.
        std::vector<cv::Point2f> movingPoints;
        int survivedFeatures = 0;
        for (size_t i = 0; i < nextPoints.size(); ++i)
        {
            if (!status[i])
            {
                continue;
            }

            float dx = nextPoints[i].x - activePoints[i].x;
            float dy = nextPoints[i].y - activePoints[i].y;
            float motion = std::sqrt(dx * dx + dy * dy);

            if (motion > minMovement && motion < maxMovement)
            {
                movingPoints.push_back(nextPoints[i]);
                ++survivedFeatures;

                if (logFeatureMotion)
                {
                    std::cout << "[DEBUG][Frame " << (frameCounter - 1) << "->" << frameCounter << "] feature " << i
                              << " prev=(" << activePoints[i].x << "," << activePoints[i].y << ")"
                              << " next=(" << nextPoints[i].x << "," << nextPoints[i].y << ")"
                              << " motion=" << motion << std::endl;
                }
            }
        }

        if (logFrameSummary)
        {
            std::cout << "[DEBUG][Frame " << (frameCounter - 1) << "->" << frameCounter << "] sopravvissute "
                      << survivedFeatures << " su " << nextPoints.size() << std::endl;
        }

        cv::Rect currentBox;
        if (computeBoundingBoxFromPoints(movingPoints, currentFrame.size(), currentBox))
        {
            // Aggiorna la bbox solo quando la stima corrente e' valida.
            lastBox = currentBox;
            hasLastBox = true;
        }

        std::vector<cv::Point2f> nextActivePoints = movingPoints;

        if (frameCounter % refreshEveryNFrames == 0)
        {
            // Reiniezione periodica di nuove feature per ridurre deriva/perdite.
            std::vector<cv::Point2f> refreshedPoints;
            detectSIFTPoints(currentGray, sift, refreshedPoints);
            nextActivePoints.insert(nextActivePoints.end(), refreshedPoints.begin(), refreshedPoints.end());
        }

        featureList trackedFeatures;
        for (size_t i = 0; i < movingPoints.size(); ++i)
        {
            trackedFeatures.addFeature(cv::Point(
                static_cast<int>(movingPoints[i].x),
                static_cast<int>(movingPoints[i].y)
            ));
        }

        std::vector<cv::Point> integerPoints = trackedFeatures.getFeaturePoints();

        // Salva frame annotato con la bbox piu' recente disponibile.
        cv::Mat output = currentFrame.clone();

        cv::rectangle(output, lastBox, cv::Scalar(0, 255, 0), 2);

        if (showSavedFeatures)
        {
            // Visualizza sul frame corrente le feature mantenute dopo i filtri.
            for (size_t i = 0; i < integerPoints.size(); ++i)
            {
                cv::circle(output, integerPoints[i], 2, cv::Scalar(0, 0, 255), -1);
            }
        }

        std::string outputPath = outputFolder + cv::format("/frame_%04d.png", frameCounter);
        if (!cv::imwrite(outputPath, output))
        {
            std::cerr << "Errore: impossibile salvare " << outputPath << std::endl;
            return false;
        }

        activePoints = nextActivePoints;
        prevGray = currentGray.clone();
        ++frameCounter;
    }

    if (logCategorySummary)
    {
        std::cout << "[DEBUG] Fine categoria " << inputFolder << ", frame elaborati: " << frameCounter << std::endl;
    }

    std::cout << "Elaborazione completata. Immagini annotate salvate in: " << outputFolder << std::endl;
    return true;
}

int main(int argc, char** argv){
    // Modalita' 1: una singola cartella passata da riga di comando.
    if (argc == 2)
    {
        return processCategory(argv[1]) ? 0 : 1;
    }

    // Modalita' 2: senza argomenti, elabora in sequenza tutte le categorie del dataset.
    if (argc > 2)
    {
        std::cerr << "Uso: " << argv[0] << " [path_cartella_immagini]" << std::endl;
        return 1;
    }

    std::vector<std::string> categories;
    categories.push_back("dataset/data/bird");
    categories.push_back("dataset/data/car");
    categories.push_back("dataset/data/frog");
    categories.push_back("dataset/data/sheep");
    categories.push_back("dataset/data/squirrel");

    // Log esplicito di avanzamento tra categorie per tracciare l'esecuzione batch.
    for (size_t i = 0; i < categories.size(); ++i)
    {
        std::cout << "[Categoria " << (i + 1) << "/" << categories.size() << "] Inizio elaborazione: "
                  << categories[i] << std::endl;

        if (!processCategory(categories[i]))
        {
            std::cerr << "Errore durante l'elaborazione della categoria " << (i + 1) << std::endl;
            return 1;
        }

        if (i + 1 < categories.size())
        {
            std::cout << "Passaggio da elaborazione categoria " << (i + 1)
                      << " a elaborazione categoria " << (i + 2) << std::endl;
        }
    }

    std::cout << "Elaborazione completata per tutte le categorie." << std::endl;
    return 0;
}