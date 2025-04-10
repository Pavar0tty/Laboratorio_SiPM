#!/usr/bin/env python3
import csv
import argparse

def convert_value(value):
    """
    Se il valore contiene una virgola come separatore decimale e rappresenta un numero,
    converte la stringa in un numero float e poi la riporta come stringa con il punto.
    Se non è un numero, restituisce il valore originale.
    """
    # Se la cella è vuota, la restituisco invariata.
    if value.strip() == "":
        return value
    try:
        # Tenta di convertire sostituendo la virgola con il punto
        num = float(value.replace(',', '.'))
        # Restituisce il numero in formato stringa (con punto decimale)
        return str(num)
    except ValueError:
        # Se la conversione fallisce, il campo non è un numero puro: restituisce il valore originale.
        return value

def convert_csv(input_file, output_file):
    """
    Legge un file CSV con delimitatore ';' e per ogni cella prova a convertire
    i numeri dal formato italiano (con virgola) a quello internazionale (con punto).
    Scrive il risultato in un nuovo file CSV con delimitatore ','.
    """
    with open(input_file, newline='', encoding='utf-8') as csv_in:
        reader = csv.reader(csv_in, delimiter=';')
        with open(output_file, 'w', newline='', encoding='utf-8') as csv_out:
            writer = csv.writer(csv_out, delimiter=',')
            for row in reader:
                new_row = [convert_value(cell) for cell in row]
                writer.writerow(new_row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converti file CSV esportati da Excel (italiano) in un formato standard.")
    parser.add_argument("input", help="Percorso del file CSV in input")
    parser.add_argument("output", help="Percorso del file CSV in output")
    args = parser.parse_args()
    
    convert_csv(args.input, args.output)
