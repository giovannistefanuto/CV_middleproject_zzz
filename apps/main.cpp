#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <utils.h>
    
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