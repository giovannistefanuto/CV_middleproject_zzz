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
#include "utils.h"

static bool processCategory(const std::string& inputFolder)
{
    // Ogni categoria salva output in una cartella dedicata accanto all'input.

    const int refreshEveryNFrames = 2;
    const bool showSavedFeatures = true;

    bool logFeatureMotion=keepDebugOutput(inputFolder);

    const bool logFrameSummary = logFeatureMotion;
    const bool logCategorySummary = logFeatureMotion;

    // Iteratore immagini della categoria (supporta PNG/JPG ordinati).
    ImageIterator iterator(inputFolder);
    if (!iterator.hasNext()) {
        std::cerr << "Errore: nessuna immagine PNG/JPG trovata in " << inputFolder << std::endl;
        return false;
    }

    int aliveFeatures, numFrames=15;
    std::string category= getLastPart(inputFolder);
    std::vector<cv::Mat> frames(numFrames);
    std::vector<float> motions, err;
    std::vector<uchar> status;
    std::vector<cv::Point2f> referencePoints, movedPoints, printablePoints;
    std::vector<cv::Point> realPoints = extract_ground_truth(category);

    cv::Rect box;
    cv::Mat referenceGray, actualGray;
    cv::Point box_point_1,box_point_2;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(60000);

    for(int i=0;iterator.hasNext();i++)
    {
        iterator.next(frames[i%numFrames]);
        if(i%numFrames==0)
        {
            cv::cvtColor(frames[0], referenceGray, cv::COLOR_BGR2GRAY);
            referencePoints.clear();
            detectSIFTPoints(referenceGray, sift, referencePoints);
            motions = std::vector<float>(referencePoints.size());
            continue;
        }
        cv::cvtColor(frames[i%numFrames], actualGray, cv::COLOR_BGR2GRAY);
        cv::calcOpticalFlowPyrLK(referenceGray, actualGray, referencePoints, movedPoints, status, err);
        accumulateMotion(movedPoints,referencePoints,status,motions);
        if((i+1)%numFrames==0 || !iterator.hasNext())
        {
            aliveFeatures = featureFilter(referencePoints,printablePoints,motions);
            computeBoundingBoxFromPoints(printablePoints, frames[0].size(), box);
            box_point_1= cv::Point(box.x,box.y);
            box_point_2= cv::Point(box.x+box.width,box.y+box.height);
            std::vector<cv::Point> boxPoints{box_point_1,box_point_2};

            if(i==numFrames-1)
            {
                float score=evaluate_mIoU(boxPoints,realPoints);
                std::cout<<score<<std::endl;
            }
            bool result=true;
            for(int j=0;j<numFrames;j++)
            {   
                result &= saveFrame(inputFolder,frames[j],box,printablePoints,i-(numFrames-j)+1,showSavedFeatures);
            }
            if(!result)
            {
                return false;
            }
        }
    }
    return true;
}

    
    // Salvataggio del primo frame annotato.
    
    

    /*
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
    std::vector<float> motions;

    cv::Mat pendingFrame;
    bool hasPendingFrame = false;

    // Prima box: uso Optical Flow tra frame 0 e frame 1 per massima precisione sul frame iniziale.
    if (iterator.hasNext() && !activePoints.empty())
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

            // Stima del moto delle feature dal primo al secondo frame.
            cv::calcOpticalFlowPyrLK(prevGray, pendingGray, activePoints, firstToSecondPoints, firstStatus, firstErr);
            motions=std::vector<float>(activePoints.size());

            accumulateMotion(firstToSecondPoints,activePoints,firstStatus,motions);
            
            cv::Mat relevantFrame=pendingFrame.clone();
            iterator.next(pendingFrame);//TODO

            int survivedInFirstFlow = featureFilter(activePoints,firstSavedPoints,motions);

            
            if (logFrameSummary)
            {
                std::cout << "[DEBUG][Frame 0->1] sopravvissute " << survivedInFirstFlow
                          << " su " << firstToSecondPoints.size() << std::endl;
            }
            
            if (computeBoundingBoxFromPoints(firstSavedPoints, firstFrame.size(), lastBox))
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

    cv::Point box_point_1(lastBox.x,lastBox.y);
    cv::Point box_point_2(lastBox.x+lastBox.width,lastBox.y+lastBox.height);

    std::vector<cv::Point> boxPoints{box_point_1,box_point_2}; 

    std::string category= getLastPart(inputFolder);
    std::vector<cv::Point> realPoints = extract_ground_truth(category);
    float score=evaluate_mIoU(boxPoints,realPoints);
    // Salvataggio del primo frame annotato.
    if(!saveFrame(inputFolder,firstFrame,lastBox,firstSavedPoints,0,showSavedFeatures))
    {
        return false;
    }
    return true;
*/
/*
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
        if(frameCounter%5==0){
            motions=std::vector<float>(activePoints.size());
        }

        int survivedFeatures = featureFilter(nextPoints,activePoints,status,movingPoints,motions,logFeatureMotion);

        /*if (logFrameSummary)
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

        if(!saveFrame(inputFolder,currentFrame,lastBox,movingPoints,frameCounter,showSavedFeatures))
        {
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

    std::cout << "Elaborazione completata. Immagini annotate salvate in: " << inputFolder << " followed by the desired words" << std::endl;
    std::cout << "ACTUAL SCORE: "<<score<<std::endl;
    return true;
    */
//}

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
    categories.push_back("../dataset/data/bird");
    categories.push_back("../dataset/data/car");
    categories.push_back("../dataset/data/frog");
    categories.push_back("../dataset/data/sheep");
    categories.push_back("../dataset/data/squirrel");

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